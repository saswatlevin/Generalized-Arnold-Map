all:program

program:kernel.o
	g++ -o program -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart main1.cpp kernel.o 

kernel.o:
	nvcc -c -arch=sm_30 kernel.cu 

clean:rm -f *.o program
