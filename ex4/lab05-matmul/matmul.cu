// adapted from https://github.com/Kaixhin/cuda-workshop

#include "cuda_helper.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <cassert>

enum class Implementation {
  CPU, CUDA_SIMPLE, CUDA_TILED
};

// -- CPU part --
//
// NOTE:
// - the matrices are flat 1D arrays, so a mapping from a 2D to 1D index is required
// - A, B, C are initialized in the main function
// - these matmul_* functions are called several times for benchmarking
//
// Template parameter T is the placeholder data type of the matrix elements (e.g. int or float)
template<typename T>
void matmul_cpu(T * const c,
                T const * const a,
                T const * const b,
                const int width) {

// CPU matmul code, using ikj loop structure
// - the most inner loop computes an axpy operation
// - ikj loop performs best on the CPU (with enabled compiler optimizations)

  // TODO: for (...
  for(int row = 0; row < width; row++){
    for(int k = 0; k < width; k++) {
      for(int col = 0; col < width; col++){
        c[row*width + col] += a[row*width + k] * b[k*width + col];

      }
    }
  }
  
}

// -- GPU part

// NOTE:
// - take the loop structure from the CPU implementation,
// and apply grid-striding in 2D:
// e.g., in the row loop, row++ becomes row += blockDim.y * gridDim.y
// - to reduce global memory accesses (writes are not cached on GPU until Volta),
// use an ijk order and a register variable to sum up the dot product
// - afer loop k is finished, store the result to C_ij
template<typename T>
__global__
void matmul_simple(T * const c,
                   T const * const a,
                   T const * const b,
                   const int width) {

  // TODO: for (...
  for(int row = threadIdx.y + blockIdx.y * blockDim.y;
          row < width;
          row += blockDim.y * gridDim.y){

            for(int col = threadIdx.x + blockIdx.x * blockDim.x;
                    col < width;
                    col += blockDim.x * gridDim.x){

                      //define here a result variable
                      T result = 0;

                      for(int k =0; k < width; k++){
                        result += a[row*width + k] * b[k*width + col];
                      }
              }
    }
}

// NOTE:
// - only squared dim allowed (width*width)
// - only blockdim = Tilewidth*Tilewidth allowed
// - only dim = x * Tilewidth allowed (multiples of Tilewidth)
// Take the loop structure from the simple GPU kernel and
// split up the k-loop into two loops:
// - loop through the tiles 0,..,nr_tiles_x-1
//   - load block of A into s_a, and block of B into s_b
//   - __syncthreads()
//   - loop to compute the partial dot product
//     - temp += s_a[threadIdx.y][·] * s_b[·][threadIdx.x]
//   - __syncthreads()
template<int Tilewidth, typename T>
__global__ void matmul_tiled(T * const c,
                             T const * const a,
                             T const * const b,
                             const int width) {

// GPU matmul code, using static shared memory
// (2D array, Tilewidth x Tilewidth)

  // Allocate 2D tiles in shared memory
  __shared__ T s_a[Tilewidth][Tilewidth];
  __shared__ T s_b[Tilewidth][Tilewidth];

  // nr_tiles_x -> no. of phases
  const int nr_tiles_x = width/Tilewidth;

  // TODO: for (...
  for(int row = threadIdx.y + blockIdx.y * blockDim.y;
          row < width;
          row += blockDim.y * gridDim.y){

            for(int col = threadIdx.x + blockIdx.x * blockDim.x;
                    col < width;
                    col += blockDim.x * gridDim.x){

                      //define here a result variable
                      T result = 0;

                      // fetch tile matrix into shared memory in phases
                      for(int p = 0; p < nr_tiles_x; p++){
                        s_a[threadIdx.y][threadIdx.x] = a[row*width + (p*Tilewidth + threadIdx.x)];
                        s_b[threadIdx.y][threadIdx.x] = a[(p*Tilewidth + threadIdx.x)*width + col];
                      }
                      __syncthreads());
                      // This is an operation only on data in the shared variables s_a and s_b
                      for(int ti = 0; ti < Tilewidth; ti++){
                        s_a[threadIdx.y][threadIdx.x] = a[row*width + (p*Tilewidth + threadIdx.x)];
                        s_b[threadIdx.y][threadIdx.x] = a[(p*Tilewidth + threadIdx.x)*width + col];
                        result += s_a[threadIdx.y][ti] * s_b[ti][threadIdx.x]];
                      }
                      __syncthreads();
              }
              c[row*width +col = result];
    }
}

// -- ignore code below

// calls CPU implementation of matmul
template<typename T,
         int TRepetitions
         >
double matmul(
  T* const h_matrix_a,
  T* const h_matrix_b,
  T* const h_matrix_c,
  size_t width) {

  double milliseconds = 0;
  double min_ms = std::numeric_limits<double>::max();

  // -- REPETITIONS --
  // Take the minimum of all measured runtimes.
  // Note: For reduction of the overhead part (~µs), multiple repetitions r2 within
  // the time measurement can be considered as well (avg. runtime /= r2).
  for(int r=0; r<TRepetitions; ++r) {

    TimerCPU timerCPU;

    memset(h_matrix_c, 0, width*sizeof(T));

    timerCPU.startTimer(); // see cuda_helper.cuh

    matmul_cpu(
      h_matrix_c,
      h_matrix_a,
      h_matrix_b,
      width
      );

    milliseconds = timerCPU.stopTimer();

    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  return min_ms;
}

template<Implementation TImpl,
         typename T,
         int TRepetitions,
         int TBlocksizeX = 16,
         int TTilewidthX = TBlocksizeX
         >
double matmul(
        T* const h_matrix_a,
        T* const h_matrix_b,
        T* const h_matrix_c,
        const size_t width,
        const int dev,
        const int blocksPerSM) {
  static_assert(TImpl!=Implementation::CPU, "Only for GPU kernels.");

  CHECK_CUDA( cudaSetDevice(dev) );
  CHECK_CUDA( cudaFree(0) ); // force context init

  cudaEvent_t cstart, cend;
  cudaStream_t cstream;
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cend));
  CHECK_CUDA(cudaStreamCreate(&cstream));

  std::size_t n = width*width; // square matrix
  std::size_t n_bytes = n * sizeof(T);

  // Allocation on device
  T* d_matrix_a = 0;
  T* d_matrix_b = 0;
  T* d_matrix_c = 0;
  CHECK_CUDA( cudaMalloc(&d_matrix_a, n_bytes) );
  CHECK_CUDA( cudaMalloc(&d_matrix_b, n_bytes) );
  CHECK_CUDA( cudaMalloc(&d_matrix_c, n_bytes) );

  // Copy to device

  CHECK_CUDA( cudaMemcpy( d_matrix_a, h_matrix_a, n_bytes, cudaMemcpyHostToDevice) );
  CHECK_CUDA( cudaMemcpy( d_matrix_b, h_matrix_b, n_bytes, cudaMemcpyHostToDevice) );
  CHECK_CUDA( cudaMemcpy( d_matrix_c, h_matrix_c, n_bytes, cudaMemcpyHostToDevice) );

  // number of SMs, needed for 2D grid-striding loop. For one dimension use ceil(sqrt(numSMs)).
  int numSMs;
  CHECK_CUDA( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev) );
  int sqrtNumSMs = std::ceil(std::sqrt(static_cast<double>(numSMs)));

  // TBlocksize would be TBlocksizeX^2.
  dim3 blocksize(TBlocksizeX, TBlocksizeX);
  dim3 blocks_n((width-1)/TBlocksizeX+1, (width-1)/TBlocksizeX+1); // for monolithic kernels
  dim3 blocks_i(sqrtNumSMs, sqrtNumSMs);
  assert(blocksize.x == TTilewidthX);
  assert(blocksize.y == TTilewidthX);

  // due to square 2D sqrt is needed: sqrt(blocksPerSM) * sqrt(blocksPerSM) = blocksPerSM
  blocks_i.x = std::floor(std::sqrt(static_cast<double>(blocksPerSM))*blocks_i.x+0.5);
  blocks_i.y = std::floor(std::sqrt(static_cast<double>(blocksPerSM))*blocks_i.y+0.5);
  if(blocks_i.x>blocks_n.x)
    blocks_i.x = blocks_n.x;
  if(blocks_i.y>blocks_n.y)
    blocks_i.y = blocks_n.y;

  float milliseconds = 0;
  float min_ms = std::numeric_limits<float>::max();

  // -- REPETITIONS --
  // Take the minimum of all measured runtimes.
  // Note: For reduction of the overhead part (~µs), multiple repetitions r2 within
  // the time measurement can be considered as well (avg. runtime /= r2).
  for(int r=0; r<TRepetitions; ++r) {

    // CHECK_CUDA( cudaMemset( d_matrix_c, 0, n_bytes) ); // if you want to initialize C to zero in each step

    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_CUDA(cudaEventRecord(cstart, cstream));

    switch(TImpl) {
    case Implementation::CUDA_SIMPLE:
      matmul_simple<<<blocks_i, blocksize, 0, cstream>>>(
        d_matrix_c,
        d_matrix_a,
        d_matrix_b,
        width
        );
      break;
    case Implementation::CUDA_TILED:
      matmul_tiled<TTilewidthX><<<blocks_i, blocksize, 0, cstream>>>(
        d_matrix_c,
        d_matrix_a,
        d_matrix_b,
        width
        );
      break;
    }

    CHECK_CUDA( cudaGetLastError() );

    CHECK_CUDA( cudaEventRecord(cend, cstream) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );

    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  // -- check GPU results (of the last repetition)
  CHECK_CUDA( cudaMemcpy( h_matrix_c, d_matrix_c, n_bytes, cudaMemcpyDeviceToHost) );
  T* h_matrix = new T[n];
  memset(h_matrix, 0, n_bytes);
  matmul_cpu(h_matrix, h_matrix_a, h_matrix_b, width);

  std::cout << "Validation...\n";
  for(size_t j=0; j<n; ++j) {
    if( h_matrix[j] != h_matrix_c[j] ) {
      std::cerr << "\n\n" << h_matrix_c[j] << " != " << h_matrix[j] << " [i=" <<j<<"]\n";
      break;
    }
  }
  std::cout << "Validation: Done.\n";
  delete[] h_matrix;

  CHECK_CUDA(cudaFree(d_matrix_a));
  CHECK_CUDA(cudaFree(d_matrix_b));
  CHECK_CUDA(cudaFree(d_matrix_c));
  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));
  CHECK_CUDA(cudaStreamDestroy(cstream));

  return static_cast<double>(min_ms);
}

// matmul
int main(int argc, const char** argv)
{
  // number of repetitions for benchmark
  static constexpr int REPETITIONS = 3; //5;
  // data type of matrix elements
  using TMatrixType = int;


  int mode = 0; // 0 = simple, 1 = tiled, 2 = cpu
  int width = 0;
  int blocksPerSM = 0;

  if(argc >= 2)
    width = atoi(argv[1]);
  if(argc >= 3) // mode
    mode = atoi(argv[2]);
  if(argc >= 4) // blocksPerSM
    blocksPerSM = atoi(argv[3]);

  if(width<2)
    width = 1<<4;
  if(blocksPerSM<1)
    blocksPerSM = 1;

  std::cout << "USAGE: ./matmul [width] [mode] [blocksPerSM]\n";
  std::cout << "VALUES: " << argv[0]
            << " width=" << width
            << " mode=" << mode
            << " blocksPerSM=" << blocksPerSM
    ;

  if(mode==0)
    std::cout << " (matmul-cpu)";
  else if(mode==1)
    std::cout << " (matmul-simple)";
  else if(mode==2)
    std::cout << " (matmul-tiled)";
  else {
    std::cout << "\nUnknown mode.\n";
    return EXIT_FAILURE;
  }
  std::cout << "\n\n";

  // -- device information

  const int dev=0;
  std::cout << getCUDADeviceInformations(dev).str() << "\n\n";

  // -- host data allocation

  std::size_t n = width*width; // square matrix
  std::size_t n_bytes = n * sizeof(TMatrixType);
  TMatrixType* h_matrix_a = new TMatrixType[n];
  TMatrixType* h_matrix_b = new TMatrixType[n];
  TMatrixType* h_matrix_c = new TMatrixType[n];

  // -- init host data

  std::srand(1337);
  for (size_t i = 0; i < n; i++) {
    h_matrix_a[i] = std::rand()/((RAND_MAX + 1u)/6);  // Note: 1+rand()%6 is biased
    h_matrix_b[i] = std::rand()/((RAND_MAX + 1u)/6);  // Note: 1+rand()%6 is biased
    h_matrix_c[i] = 0;
  }

  double min_ms_cpu = std::numeric_limits<double>::max();
  double min_ms_16  = std::numeric_limits<double>::max();
  double min_ms_32  = std::numeric_limits<double>::max();
  try{

    if(mode==0) { // CPU

      // Mode, T, TRepetitions, TBlocksizeX[, TTilesizeX=TBlocksizeX]
      min_ms_cpu = matmul<TMatrixType, REPETITIONS>(
        h_matrix_a,
        h_matrix_b,
        h_matrix_c,
        width);

    } else if(mode==1) { // SIMPLE

      min_ms_16 = matmul<Implementation::CUDA_SIMPLE, TMatrixType, REPETITIONS, 16>(
        h_matrix_a,
        h_matrix_b,
        h_matrix_c,
        width,
        dev,
        blocksPerSM);
      min_ms_32 = matmul<Implementation::CUDA_SIMPLE, TMatrixType, REPETITIONS, 32>(
        h_matrix_a,
        h_matrix_b,
        h_matrix_c,
        width,
        dev,
        blocksPerSM);

    } else if (mode==2) { // TILED

      min_ms_16 = matmul<Implementation::CUDA_TILED, TMatrixType, REPETITIONS, 16>(
        h_matrix_a,
        h_matrix_b,
        h_matrix_c,
        width,
        dev,
        blocksPerSM);
      min_ms_32 = matmul<Implementation::CUDA_TILED, TMatrixType, REPETITIONS, 32>(
        h_matrix_a,
        h_matrix_b,
        h_matrix_c,
        width,
        dev,
        blocksPerSM);
    }

    double min_ms;
    if( mode==0 ) {
      min_ms = min_ms_cpu;
    } else if( min_ms_16 < min_ms_32 ) {
      std::cout << "Most performant: 16x16 Threads per block (tilesize=blocksize)";
      min_ms = min_ms_16;
    } else {
      std::cout << "Most performant: 32x32 Threads per block (tilesize=blocksize)";
      min_ms = min_ms_32;
    }
    std::cout << "\n Min. time: " << min_ms << " ms"
              << "\n Memory: " << 3*n_bytes/min_ms*1e-6 << " GB/s"
              << "\n Compute: " << 2.0*n*1e-6*width/min_ms << " GFLOP/s" // 2*width^3 ops
              << "\n";

  }catch(std::runtime_error e){
    std::cerr << "\n" << e.what() << "\n";
    CHECK_CUDA( cudaDeviceReset() );
    return EXIT_FAILURE;
  }

  delete[] h_matrix_a;
  delete[] h_matrix_b;
  delete[] h_matrix_c;
  CHECK_CUDA( cudaDeviceReset() );
  return EXIT_SUCCESS;
}
