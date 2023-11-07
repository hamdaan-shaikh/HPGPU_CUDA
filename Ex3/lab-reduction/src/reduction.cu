#include "cuda_helper.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <limits>

template<int TBlocksize, typename T>
__global__
void kernel_reduce(T* x, T* y, int n)
{
  
  // TODO:
  // with Global memory
  // - each thread starts with x[i] and sums it up with x[i+grid], ...
  // - atomicAdd the results into y

  // with Shared Memory and without atomics
  // - sum up the elements stored in x
  // and compute the partial results as done in the first task with global memory
  // - store it into shared memory
  // - perform a block and warp reduce (can be described as one loop)
  // - either use atomicAdd or a second pass, where block results are summed up

  // --------
  // Level 1: grid reduce
  // --------

  // --------
  // Level 2: block + warp reduce (on shared memory)
  // --------
  // you might want to assume TBlocksize to be power-of-2 to save some checks


  // TODO:
  // store block result to gmem (only one thread per block should do this)
}

template<typename T, int TRuns, int TBlocksize>
void reduce(T init, size_t n, int dev) {

  CHECK_CUDA( cudaSetDevice(dev) );
  cudaDeviceProp prop;
  CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
  cudaEvent_t cstart, cend;
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cend));

  std::cout << getCUDADeviceInformations(dev).str()
            << "\n\n";

  // for grid-striding loops get number of SMs
  int numSMs;
  CHECK_CUDA( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev) );
  dim3 blocks( 16*numSMs ); // 16*128 = 2048 (=max resident threads on SM), rule of thumb
  if( blocks.x > ((n-1)/TBlocksize+1) )
    blocks.x = (n-1)/TBlocksize+1;

  T* h_x = new T[n]; // [on host]
  T* x = nullptr; // input data [on device]
  T* y = nullptr; // result [on device]
  T result_gpu = 0; // final result [on host]

  // TODO: allocate memory
  // We allocate the host size of n for x, and unit size for y
  CHECK_CUDA ( cudaMalloc(&x,n*sizeof(T)) ); 
  CHECK_CUDA ( cudaMalloc(&y,sizeof(T)) );

  // init host memory
  for (int i = 0; i < n; i++) {
    h_x[i] = init;
  }

  // TODO: transfer data to GPU
  CHECK_CUDA ( cudaMemcpy(x, h_x, n*sizeof(T), cudaMemcpyHostToDevice) );

  // time measurement
  float milliseconds = 0;
  float min_ms = std::numeric_limits<float>::max();

  for(int r=0; r<TRuns; ++r) {
    CHECK_CUDA ( cudaMemset( y, 0 , sizeof(T) ) );
    CHECK_CUDA(cudaEventRecord(cstart));

    // TODO: call the kernel (maybe second pass needed for summing up partial results)
    //  kernel_reduce<TBlocksize><<<blocks, TBlocksize>>>( ...
    // Attention: do not write to x as we call this part several times for benchmark
    

    CHECK_CUDA( cudaEventRecord(cend) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  // TODO: get the result
  CHECK_CUDA ( cudaMemcpy(y, x, sizeof(T), cudaMemcpyHostToDevice) );
  CHECK_CUDA ( cudaMemcpy(result_gpu, y, sizeof(T), cudaMemcpyDeviceToHost) );

  // output
  std::cout << "Result (n = "<<n<<"):\n"
            << "GPU: " << result_gpu << " (min kernels time = "<< min_ms <<" ms)\n"
            << "expected: " << init*n <<"\n"
            << (init*n != result_gpu ? "MISMATCH!!" : "Success") << "\n"
            << "max bandwidth: "<<n*sizeof(T)/min_ms*1e-6<<" GB/s"
            << std::endl;

  delete[] h_x;
  CHECK_CUDA(cudaFree(x));
  CHECK_CUDA(cudaFree(y));
  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));
}

int main(int argc, const char** argv)
{
  int dev=0;
  int n = 0;
  if(argc==2)
    n = atoi(argv[1]);
  if(n<2)
    n = 1<<28; // = 2^28
  try{
    // reduce<data type, number of benchmark runs, number of threads per block>(init, n, dev_id)
    reduce<int, 5, 128>(1, n, dev);
  }catch(std::runtime_error& e){
    std::cerr << e.what() << "\n";
    CHECK_CUDA( cudaDeviceReset() ); // always call this at the end of your CUDA program
    return 1;
  }
  CHECK_CUDA( cudaDeviceReset() ); // always call this at the end of your CUDA program
  return 0;
}
