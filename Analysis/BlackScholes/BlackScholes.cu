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

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */


#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    double *h_CallResult,
    double *h_PutResult,
    double *h_StockPrice,
    double *h_OptionStrike,
    double *h_OptionYears,
    double Riskfree,
    double Volatility,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"
#include "BlackScholes_kernel_sf.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random double in [low, high] range
////////////////////////////////////////////////////////////////////////////////
double RandFloat(double low, double high)
{
    double t = (double)rand() / (double)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 4000000;
const int  NUM_ITERATIONS = 1024;


const int          OPT_SZ = OPT_N * sizeof(double);
const double      RISKFREE = 0.02f;
const double    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Start logs
    printf("[%s] - Starting...\n", argv[0]);

    //'h_' prefix - CPU (host) memory space
    double
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU copy of GPU results, single precision
    *h_CallResultGPU_sf,
    *h_PutResultGPU_sf,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    double
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //Results calculated by GPU, single precision
    *d_CallResult_sf,
    *d_PutResult_sf,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    double
    delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

    StopWatchInterface *hTimer = NULL;
    int i;

    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (double *)malloc(OPT_SZ);
    h_PutResultCPU  = (double *)malloc(OPT_SZ);
    h_CallResultGPU = (double *)malloc(OPT_SZ);
    h_PutResultGPU  = (double *)malloc(OPT_SZ);
    h_CallResultGPU_sf = (double *)malloc(OPT_SZ);
    h_PutResultGPU_sf  = (double *)malloc(OPT_SZ);
    h_StockPrice    = (double *)malloc(OPT_SZ);
    h_OptionStrike  = (double *)malloc(OPT_SZ);
    h_OptionYears   = (double *)malloc(OPT_SZ);

    printf("...allocating GPU memory for options.\n");
    checkCudaErrors(cudaMalloc((void **)&d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_CallResult_sf,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult_sf,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears,  OPT_SZ));

    printf("...generating input data in CPU mem.\n");
    srand(5347);

    //Generate options set
    for (i = 0; i < OPT_N; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }

    srand(time(NULL));
#if 1
    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
    printf("Data init done.\n\n");
#endif

#if 1
#if 0
    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
    printf("Data init done.\n\n");
#endif

    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    checkCudaErrors(cudaDeviceSynchronize());

    puts("\nStarting double precision calculation.");
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        BlackScholesGPU<<<DIV_UP(OPT_N, 128), 128/*480, 128*/>>>(
            d_CallResult,
            d_PutResult,
            d_StockPrice,
            d_OptionStrike,
            d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );
        getLastCudaError("BlackScholesGPU() execution failed\n");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;

    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
    printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(double)) * 1E-9) / (gpuTime * 1E-3));
    printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

    printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
           (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);
#endif

#if 0
#if 0
    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
    printf("Data init done.\n\n");
#endif

    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    checkCudaErrors(cudaDeviceSynchronize());

    puts("\nStarting single precision calculation.");
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        BlackScholesGPU_sf<<<DIV_UP(OPT_N, 128), 128/*480, 128*/>>>(
            d_CallResult_sf,
            d_PutResult_sf,
            d_StockPrice,
            d_OptionStrike,
            d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );
        getLastCudaError("BlackScholesGPU_sf() execution failed\n");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;

    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
    printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(double)) * 1E-9) / (gpuTime * 1E-3));
    printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

    printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
           (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);
#endif

    printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
    checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));
#if 0
    checkCudaErrors(cudaMemcpy(h_CallResultGPU_sf, d_CallResult_sf, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU_sf,  d_PutResult_sf,  OPT_SZ, cudaMemcpyDeviceToHost));
#endif


    printf("Checking the results...\n");
    printf("...running CPU calculations.\n\n");
    //Calculate options values on CPU
    BlackScholesCPU(
        h_CallResultCPU,
        h_PutResultCPU,
        h_StockPrice,
        h_OptionStrike,
        h_OptionYears,
        RISKFREE,
        VOLATILITY,
        OPT_N
    );

#if 1
    printf("Comparing the results (CPU double - GPU double)...\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (i = 0; i < OPT_N; i++)
    {
        ref   = h_CallResultCPU[i];
        delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
    printf("L1 norm: %E\n", L1norm);
    printf("Max absolute error: %E\n\n", max_delta);
#endif

#if 0
    printf("Comparing the results (CPU double - GPU float)...\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (i = 0; i < OPT_N; i++)
    {
        ref   = h_CallResultCPU[i];
        delta = fabs(h_CallResultCPU[i] - h_CallResultGPU_sf[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
    printf("L1 norm: %E\n", L1norm);
    printf("Max absolute error: %E\n\n", max_delta);
#endif

#if 0
    printf("Comparing the results (GPU double - GPU float)...\n");
    //Calculate max absolute difference and L1 distance
    //between GPU and GPU_sf results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (i = 0; i < OPT_N; i++)
    {
        ref   = h_CallResultGPU[i];
        delta = fabs(h_CallResultGPU[i] - h_CallResultGPU_sf[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
    printf("L1 norm: %E\n", L1norm);
    printf("Max absolute error: %E\n\n", max_delta);
#endif

#if 1
#if 0
    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
    printf("Data init done.\n\n");
#endif

    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    checkCudaErrors(cudaDeviceSynchronize());

    puts("\nStarting mixed precision calculations.");

    puts("P(double) | GPU Time (msec) | Memory BW (GB/s) | Gops/sec | CPU L1 norm | CPU Abs err | GPU L1 norm | GPU Abs err");
#define RAND_STEPS	10
    // First N iterations using double precision
    int prob = NUM_ITERATIONS;
start:
    // Start mixed precision calculation
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    unsigned int count = 0;
    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        if (i < prob) {
	    count++;
            BlackScholesGPU<<<DIV_UP(OPT_N, 128), 128/*480, 128*/>>>(
                d_CallResult_sf,
                d_PutResult_sf,
                d_StockPrice,
                d_OptionStrike,
                d_OptionYears,
                RISKFREE,
                VOLATILITY,
                OPT_N
            );
        } else {
            BlackScholesGPU_sf<<<DIV_UP(OPT_N, 128), 128/*480, 128*/>>>(
                d_CallResult_sf,
                d_PutResult_sf,
                d_StockPrice,
                d_OptionStrike,
                d_OptionYears,
                RISKFREE,
                VOLATILITY,
                OPT_N
            );
	}
        getLastCudaError("BlackScholesGPU() execution failed\n");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;

    //Both call and put is calculated
    printf("%g | ", (float)count / (float)NUM_ITERATIONS);
    printf("%d | ", 2 * OPT_N);
    printf("%g | ", gpuTime);
    printf("%g | ", ((double)(5 * OPT_N * sizeof(double)) * 1E-9) / (gpuTime * 1E-3));
    printf("%g | ", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

#if 0
    printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
           (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);
#endif

    //printf("\nReading back GPU results...\n");
    checkCudaErrors(cudaMemcpy(h_CallResultGPU_sf, d_CallResult_sf, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU_sf,  d_PutResult_sf,  OPT_SZ, cudaMemcpyDeviceToHost));

#if 1
    //printf("Comparing the results (CPU double - GPU mixed)...\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (i = 0; i < OPT_N; i++)
    {
        ref   = h_CallResultCPU[i];
        delta = fabs(h_CallResultCPU[i] - h_CallResultGPU_sf[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
    printf("%E | ", L1norm);
    printf("%E | ", max_delta);
#endif

#if 1
    //printf("Comparing the results (GPU double - GPU mixed)...\n");
    //Calculate max absolute difference and L1 distance
    //between GPU and GPU_sf results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (i = 0; i < OPT_N; i++)
    {
        ref   = h_CallResultGPU[i];
        delta = fabs(h_CallResultGPU[i] - h_CallResultGPU_sf[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
    printf("%E | ", L1norm);
    printf("%E\n", max_delta);
#endif

    while (prob > 0) {
	prob -= NUM_ITERATIONS / RAND_STEPS;
	goto start;
    }
#endif

    printf("Shutting down...\n");
    printf("...releasing GPU memory.\n");
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));
    checkCudaErrors(cudaFree(d_PutResult_sf));
    checkCudaErrors(cudaFree(d_CallResult_sf));

    printf("...releasing CPU memory.\n");
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU_sf);
    free(h_CallResultGPU_sf);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);
    sdkDeleteTimer(&hTimer);
    printf("Shutdown done.\n");

    printf("\n[BlackScholes] - Test Summary\n");

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    if (L1norm > 1e-6)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
