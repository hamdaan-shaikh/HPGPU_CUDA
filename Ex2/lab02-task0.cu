
// nvcc lab01.cu -o kernel -std=c++11

#define CHECK_CUDA(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

__global__ void add(int *a, int *b, int *c, int total_num_of_elements) {
/*int i =  compute global thread id */;
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i<total_num_of_elements)
  a[i]=b[i]+c[i];
}


#include <iostream>

int main(int, char**)
{
  const size_t length = 1000000;

  int* a_host = new int[length];
  int* b_host = new int[length];
  int* c_host = new int[length];
  int *a_device = 0, *b_device = 0, *c_device = 0;
  int size = length*sizeof(int);

  for(int i=0; i<length; ++i)
  {
    // initialize a_host
    a_host[i]=i;
    b_host[i]=length+i;
  }

  // allocate GPU memory on for a and b
  CHECK_CUDA ( cudaMalloc(&a_device,size) ); 
  CHECK_CUDA ( cudaMalloc(&b_device,size) );
  
  // data transfer a_host -> a_device
  CHECK_CUDA ( cudaMemcpy(a_device, a_host, size, cudaMemcpyHostToDevice) );

  // data transfer b_host -> b_device
  CHECK_CUDA ( cudaMemcpy(b_device, b_host, size, cudaMemcpyHostToDevice) );

  add<<<(length/256)+1,256>>>(c_device, b_device, a_device, length);
  
  // data transfer b_device -> b_host
  CHECK_CUDA ( cudaMemcpy(c_host, c_device, size, cudaMemcpyDeviceToHost) );
  

  // free allocated GPU memory
  CHECK_CUDA ( cudaFree(c_device) );
  CHECK_CUDA ( cudaFree(b_device) );
  CHECK_CUDA ( cudaFree(a_device) );
  CHECK_CUDA ( cudaFree(c_host) );
  CHECK_CUDA ( cudaFree(b_host) );
  CHECK_CUDA ( cudaFree(a_host) );


  //a_host[42] = 0; // provoke an error

  
  for(int i=0; i<length; ++i)
    if(c_host[i] = a_host[i] + b_host[i])
      std::cout << "Mismatch at: " << i << "\n";


  delete[] a_host;
  delete[] b_host;
  delete[] c_host;

  delete[] a_device;
  delete[] b_device;
  delete[] c_device;

  CHECK_CUDA( cudaDeviceReset() ); // needed for clean exit in profilers/debuggers
  return 0;
}
