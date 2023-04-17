all:
	nvcc hol.cu -o fig2 -O3 -arch=sm_75 -Xptxas=-v

reveng: reverse_eng.cu
	nvcc reverse_eng.cu -o reveng -O3 -arch=sm_75 -Xptxas=-v
