# The arch= clause of the -gencode= command-line option to nvcc specifies the front-end compilation target and must always be a PTX version.
# The code= clause specifies the back-end compilation target and can either be cubin or PTX or both.
# Only the back-end target version(s) specified by the code= clause will be retained in the resulting binary;
# at least one must be PTX to provide Volta compatibility
# http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

# to_string() requires C++11
# lineinfo helps to debug and profile kernels
# enables optimization and 'all' warnings on host compiler (-Xcompiler forwards flags to gcc, ..)
NVCC_OPT := -std=c++11 -lineinfo -Xcompiler -std=c++11,-Wall,-O3

# generates both SASS and PTX for specified capability.
NVCC_OPT += -arch=sm_60

# compute = virtual arch (PTX), sm = real arch (SASS).
# Creates first PTX (CC5.0). Then, SASS (CC5.0) will be created from that. PTX is discarded then.
# Only specifying code without arch is illegal (PTX CC is required to build SASS CC).
NVCC_OPT += -gencode=arch=compute_60,code=sm_60

all:
	nvcc $(NVCC_OPT) matmul.cu -o matmul
#	nvcc $(NVCC_OPT) matmul_solution.cu -o matmul_solution
#	nvcc $(NVCC_OPT) matmul_cublas.cpp -o matmul_cublas -lcublas

clean:
	rm -f matmul matmul_solution matmul_cublas
