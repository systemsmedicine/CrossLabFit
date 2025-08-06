HOST_COMPILER = gcc
GPU_ARCH = sm_80
NVCCFLAGS = -O2
LIBRARIES = -lm -lcurand
#NVCC = nvcc -ccbin $(HOST_COMPILER) $(NVCCFLAGS) -arch=$(GPU_ARCH)
NVCC = nvcc $(NVCCFLAGS) -arch=$(GPU_ARCH)

OBJS = main-qDE.o

qDE : $(OBJS)
	$(NVCC) $^ -o $@ $(LIBRARIES)

%.o : %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f *.o *~

clean-all:
	rm -f *.o *~ *.dat
