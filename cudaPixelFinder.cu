
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
extern "C"
{
#include "png_util.h"
}


__global__ void cudaMandelbrot(int Nx, int Ny, float minx, float miny, float dx, float dy, float *c_count){

  int tx = threadIdx.x; // x-coordinate in thread block
  int ty = threadIdx.y; // y-coordinate in thread block

  int bx = blockIdx.x; // x-coordinate of block
  int by = blockIdx.y; // y-coordinate of block

  int dimx = blockDim.x; // x-dimension of thread-block
  int dimy = blockDim.y; // y-dimension of thread-block

  int nx = dimx*bx + tx; // global x index of thread
  int ny = dimy*by + ty; // global y index of thread

  int n = ny*Nx + nx;

  float cx = nx*dx + minx;
  float cy = ny*dy + miny;

  float zx = 0;
  float zy = 0;
  
  int Nt = 200;
  int t, cnt=0;

  for(t=0;t<Nt;++t){
    float zxaTmp = zx*zx + cx;
    float zxTmp = -zy*zy + zxaTmp;
    float zby = zy*zx + cy;
    zy = 2.f*zby + 0.5f;
    zx = zxTmp;

    cnt += (zxTmp<4.f);
  }
  
  // now load data
  if(nx<Nx && ny<Ny){
    c_count[n] = cnt;
  }
  
}

__device__ double r(int strlen, const char *str,int n, double a, double b) {
  if (str[n % strlen] == 'A')
    return a;
  else
    return b;
}


__global__ void cudaLyapunov(int N_x, int N_y, int N, double Z_x, double Z_y, int strlen, const char *str, float *c_lyapunov){

  int tx = threadIdx.x; // x-coordinate in thread block
  int ty = threadIdx.y; // y-coordinate in thread block

  int bx = blockIdx.x; // x-coordinate of block
  int by = blockIdx.y; // y-coordinate of block

  int dimx = blockDim.x; // x-dimension of thread-block
  int dimy = blockDim.y; // y-dimension of thread-block

  int nx = dimx*bx + tx; // global x index of thread
  int ny = dimy*by + ty; // global y index of thread
  
  double x_n = 0.5;

  double a = (Z_y-Z_x) / N_x *nx + Z_x;
  double b = (Z_y-Z_x) / N_y *(N_y-ny) + Z_x;

  

  double lmb = 0.;
  #pragma unroll 50
  for(int n = 1; n <= N; n++) {
    x_n = r(strlen, str, n-1, a, b)*x_n*(1.-x_n);
    lmb += log(fabs(r(strlen, str, n, a, b)*(1.-2.*x_n)));
  }

  lmb /= N;

  // now load data

  if(nx<N_x && ny <N_y) {
    c_lyapunov[nx + ny*N_y] = (float)lmb;
  }
}


int main(int argc, char **argv){
  const int Nx = 4096;
  const int Ny = 4096;

  //const char *subfolder = "images/";
  //char baseFileName[] = "mandelbrotzoom";

  // for (int i = 0; i < 360; ++i) {
    /* box containing sample points */
   
    const float centx = 0.27934, centy = 0.053876;
    const float diam = 1;//pow(0.5, (float) i /24.);
    const float minx = centx-0.5*diam;
    const float remax = centx+0.5*diam;
    const float miny = centy-0.5*diam;
    const float immax = centy+0.5*diam;

    const float dx = (remax-minx)/(Nx-1.f);
    const float dy = (immax-miny)/(Ny-1.f);



    float *h_count = (float*) calloc(Nx*Ny, sizeof(float));
    float *c_count;

    cudaMalloc(&c_count, Nx*Ny*sizeof(float));
    
    int D = 16;
    dim3 B(D,D);
    dim3 G( (Nx+D-1)/D, (Ny+D-1)/D);

    const char *hostString = "AB";
    const char *deviceString;
    const int N = 100;
    double Zx = 1.5;
    double Zy = 4;

    cudaMalloc((void**)&deviceString, strlen(hostString) + 1);

    cudaMemcpy((void*)deviceString, hostString, strlen(hostString) + 1, cudaMemcpyHostToDevice);

    cudaEvent_t tic, toc;

    cudaEventCreate(&tic);
    cudaEventCreate(&toc);

    cudaEventRecord(tic);
    cudaLyapunov <<< G, B >>> (Nx, Ny, N, Zx, Zy, strlen(hostString), deviceString, c_count);
    //cudaMandelbrot <<< G, B >>> (Nx, Ny, minx, miny,dx, dy, c_count);

    cudaEventRecord(toc);

    cudaDeviceSynchronize(); // Wait for the kernel to finish and check for errors


    cudaMemcpy(h_count, c_count, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);

    float elapsed;
    cudaEventElapsedTime(&elapsed, tic, toc);

    elapsed /= 1000.f;
    double gbytes = Nx*Ny * 2 * sizeof(float)/1.e9;

    printf("Elapsed Kernel: %g, Achieved Bandwidth: %g\n", elapsed, gbytes/elapsed);

    //char fileName[50];  // Adjust the size as needed
    //snprintf(fileName, sizeof(fileName), "%s%s%d.png", subfolder, baseFileName, i);\\

    FILE *png = fopen("lyapunov.png", "w");
    write_gray_png(png, Nx, Ny, h_count, -5, 10);
    fclose(png);

    free(h_count);
    cudaFree(c_count);
  //}
}