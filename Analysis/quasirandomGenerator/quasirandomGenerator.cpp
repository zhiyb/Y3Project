/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// CUDA Runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "quasirandomGenerator_common.h"

////////////////////////////////////////////////////////////////////////////////
// CPU code
////////////////////////////////////////////////////////////////////////////////
extern "C" void initQuasirandomGenerator(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION]
);

extern "C" double getQuasirandomValue(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION],
    int i,
    int dim
);

extern "C" double getQuasirandomValue63(INT64 i, int dim);
extern "C" double MoroInvCNDcpu(unsigned int p);

////////////////////////////////////////////////////////////////////////////////
// GPU code
////////////////////////////////////////////////////////////////////////////////
extern "C" void initTableGPU(unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION]);
extern "C" void initTableGPU_sf(unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION]);
extern "C" void quasirandomGeneratorGPU(double *d_Output, unsigned int seed, unsigned int N);
extern "C" void inverseCNDgpu(double *d_Output, unsigned int *d_Input, unsigned int N);
extern "C" void quasirandomGeneratorGPU_sf(double *d_Output, unsigned int seed, unsigned int N);
extern "C" void inverseCNDgpu_sf(double *d_Output, unsigned int *d_Input, unsigned int N);

const int N = 1048576;

int main(int argc, char **argv)
{
    // Start logs
    printf("%s Starting...\n\n", argv[0]);

    unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION];

    double *h_OutputGPU, *d_Output;

    int dim, pos;
    double delta, ref, sumDelta, sumRef, L1norm, gpuTime;

    StopWatchInterface *hTimer = NULL;

    if (sizeof(INT64) != 8)
    {
        printf("sizeof(INT64) != 8\n");
        return 0;
    }

    cudaDeviceProp deviceProp;
    int dev = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if (((deviceProp.major << 4) + deviceProp.minor) < 0x20)
    {
        fprintf(stderr, "quasirandomGenerator requires Compute Capability of SM 2.0 or higher to run.\n");
        cudaDeviceReset();
        exit(EXIT_WAIVED);
    }

    sdkCreateTimer(&hTimer);

    printf("Allocating GPU memory...\n");
    checkCudaErrors(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * sizeof(double)));

    printf("Allocating CPU memory...\n");
    h_OutputGPU = (double *)malloc(QRNG_DIMENSIONS * N * sizeof(double));

    printf("Initializing QRNG tables...\n\n");
    initQuasirandomGenerator(tableCPU);

//#define DOUBLE
#ifdef DOUBLE
    initTableGPU(tableCPU);
#else
    initTableGPU_sf(tableCPU);
#endif

    printf("Testing QRNG...\n\n");
    checkCudaErrors(cudaMemset(d_Output, 0, QRNG_DIMENSIONS * N * sizeof(double)));
    int numIterations = 20;

    for (int i = -1; i < numIterations; i++)
    {
        if (i == 0)
        {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }
#ifdef DOUBLE
        quasirandomGeneratorGPU(d_Output, 0, N);
#else
        quasirandomGeneratorGPU_sf(d_Output, 0, N);
#endif
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer)/(double)numIterations*1e-3;
    printf("quasirandomGenerator, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers, NumDevsUsed = %u, Workgroup = %u\n",
           (double)QRNG_DIMENSIONS * (double)N * 1.0E-9 / gpuTime, gpuTime, QRNG_DIMENSIONS*N, 1, 128*QRNG_DIMENSIONS);

    printf("\nReading GPU results...\n");
    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, QRNG_DIMENSIONS * N * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Comparing to the CPU results...\n\n");
    sumDelta = 0;
    sumRef = 0;

    for (dim = 0; dim < QRNG_DIMENSIONS; dim++)
        for (pos = 0; pos < N; pos++)
        {
            ref       = getQuasirandomValue63(pos, dim);
            delta     = (double)h_OutputGPU[dim * N + pos] - ref;
            sumDelta += fabs(delta);
            sumRef   += fabs(ref);
        }

    printf("L1 norm: %E\n", sumDelta / sumRef);

    printf("\nTesting inverseCNDgpu()...\n\n");
    checkCudaErrors(cudaMemset(d_Output, 0, QRNG_DIMENSIONS * N * sizeof(double)));

    for (int i = -1; i < numIterations; i++)
    {
        if (i == 0)
        {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }

#ifdef DOUBLE
        inverseCNDgpu(d_Output, NULL, QRNG_DIMENSIONS * N);
#else
        inverseCNDgpu_sf(d_Output, NULL, QRNG_DIMENSIONS * N);
#endif
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer)/(double)numIterations*1e-3;
    printf("quasirandomGenerator-inverse, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers, NumDevsUsed = %u, Workgroup = %u\n",
           (double)QRNG_DIMENSIONS * (double)N * 1E-9 / gpuTime, gpuTime, QRNG_DIMENSIONS*N, 1, 128);

    printf("Reading GPU results...\n");
    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, QRNG_DIMENSIONS * N * sizeof(double), cudaMemcpyDeviceToHost));

    printf("\nComparing to the CPU results...\n");
    sumDelta = 0;
    sumRef = 0;
    unsigned int distance = ((unsigned int)-1) / (QRNG_DIMENSIONS * N + 1);

    for (pos = 0; pos < QRNG_DIMENSIONS * N; pos++)
    {
        unsigned int d = (pos + 1) * distance;
        ref       = MoroInvCNDcpu(d);
        delta     = (double)h_OutputGPU[pos] - ref;
        sumDelta += fabs(delta);
        sumRef   += fabs(ref);
    }

    printf("L1 norm: %E\n\n", L1norm = sumDelta / sumRef);

    printf("Shutting down...\n");
    sdkDeleteTimer(&hTimer);
    free(h_OutputGPU);
    checkCudaErrors(cudaFree(d_Output));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
}
