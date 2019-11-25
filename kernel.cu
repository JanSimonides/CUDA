#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

#define ROZMER_A 4096
#define ROZMER_B 4096


int* alokuj(int r, int s) {
	int* matica;
	matica = (int*)malloc(sizeof(int) * r * s);
	return matica;
}

void generuj(int* matica, int r, int s) {
	int i, j;
	for (i = 0; i < r; i++) {
		for (j = 0; j < s; j++) {
			matica[i * s + j] = rand() % 10;
		}
	}
}

void uvolni(int* matica) {
	cudaFree(matica);
}

void novyVypis(int* matica, int r, int s) {
	int i, j;
	for (i = 0; i < r; i++) {
		for (j = 0; j < s; j++) {
			printf("%d ", matica[i * s + j]);
		}
		putchar('\n');
	}
	putchar('\n');
}

void pocitaj(int* a, int riadkyA, int stlpceA, int* b, int riadkyB, int stlpceB) {
	int i, j, k;
	if (stlpceA != riadkyB) {
		printf("Nemozem nasobit!\n");
	}
	else {
		printf("CPU\n");
		int* vysledok = alokuj(riadkyA, stlpceB);
		int sum = 0;
		for (i = 0; i < riadkyA; i++) {
			for (j = 0; j < stlpceB; j++) {
				for (k = 0; k < stlpceA; k++) {
					sum += a[i * stlpceA + k] * b[k * stlpceB + j];
				}
				vysledok[i * stlpceB + j] = sum;
				sum = 0;
			}
		}
		//novyVypis(vysledok, riadkyA, stlpceB);
		//return vysledok;
		uvolni(vysledok);
	}
}
__global__
void gpu_pocitaj(int* a, int riadkyA, int stlpceA, int* b, int riadkyB, int stlpceB, int* c) {
	if (stlpceA != riadkyB) {
		printf("Nemozem nasobit!\n");
	}
	else {
		int sum = 0;

		int col = blockDim.x *blockIdx.x + threadIdx.x;
		int row = blockDim.y * blockIdx.y + threadIdx.y;

		int k;
		if (col < riadkyA && row < stlpceB) {
			for (k = 0; k < stlpceA; k++) {
				sum += a[col*stlpceA+k] * b[k * stlpceB + row];
			}
			c[col * stlpceB + row] = sum;
			sum = 0;
		}
	}
}

int main()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	srand(time(NULL));
	int* a, * b, * c;
	int* gpu_a, * gpu_b, * gpu_c;
	float ms = 0.0;

	a = alokuj(ROZMER_A, ROZMER_B);
	b = alokuj(ROZMER_B, ROZMER_B);
	c = alokuj(ROZMER_A, ROZMER_B);

	cudaMalloc(&gpu_a, sizeof(int) * ROZMER_A * ROZMER_B);
	cudaMalloc(&gpu_b, sizeof(int) * ROZMER_B * ROZMER_B);
	cudaMalloc(&gpu_c, sizeof(int) * ROZMER_A * ROZMER_B);

	generuj(a, ROZMER_A, ROZMER_B);
	generuj(b, ROZMER_B, ROZMER_B);

	//novyVypis(a, ROZMER_A, ROZMER_B);
	//novyVypis(b, ROZMER_B, ROZMER_B);

	cudaMemcpy(gpu_a, a, ROZMER_A * ROZMER_B * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b, ROZMER_B * ROZMER_B * sizeof(int), cudaMemcpyHostToDevice);

	uvolni(a);
	uvolni(b);

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(ROZMER_A / threadsPerBlock.x, ROZMER_B / threadsPerBlock.y);

	cudaEventRecord(start);
	gpu_pocitaj << <numBlocks, threadsPerBlock >> > (gpu_a, ROZMER_A, ROZMER_B, gpu_b, ROZMER_B, ROZMER_B, gpu_c);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	//pocitaj(a,ROZMER_A,ROZMER_B,b,ROZMER_B,ROZMER_B);

	cudaMemcpy(c, gpu_c, ROZMER_A * ROZMER_B * sizeof(int), cudaMemcpyDeviceToHost);
	//novyVypis(c, ROZMER_A, ROZMER_B);

	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);

	cudaEventElapsedTime(&ms, start, stop);

	uvolni(c);
	printf("GPU = %.6f s\n", ms/1e3);
	return 0;
}
