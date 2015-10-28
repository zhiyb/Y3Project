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



#ifndef QUASIRANDOMGENERATOR_KERNEL_CUH
#define QUASIRANDOMGENERATOR_KERNEL_CUH


#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include "quasirandomGenerator_common.h"



//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)



////////////////////////////////////////////////////////////////////////////////
// Niederreiter quasirandom number generation kernel
////////////////////////////////////////////////////////////////////////////////
static __constant__ unsigned int c_Table[QRNG_DIMENSIONS][QRNG_RESOLUTION];
// For used by single precision kernel
//__constant__ unsigned int c_Table[QRNG_DIMENSIONS][QRNG_RESOLUTION];

static __global__ void quasirandomGeneratorKernel(
    double *d_Output,
    unsigned int seed,
    unsigned int N
)
{
    unsigned int *dimBase = &c_Table[threadIdx.y][0];
    unsigned int      tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int  threadN = MUL(blockDim.x, gridDim.x);

    for (unsigned int pos = tid; pos < N; pos += threadN)
    {
        unsigned int result = 0;
        unsigned int data = seed + pos;

        for (int bit = 0; bit < QRNG_RESOLUTION; bit++, data >>= 1)
            if (data & 1)
            {
                result ^= dimBase[bit];
            }

        d_Output[MUL(threadIdx.y, N) + pos] = (double)(result + 1) * INT_SCALE;
    }
}

//Table initialization routine
extern "C" void initTableGPU(unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION])
{
    checkCudaErrors(cudaMemcpyToSymbol(
                        c_Table,
                        tableCPU,
                        QRNG_DIMENSIONS * QRNG_RESOLUTION * sizeof(unsigned int)
                    ));
}

//Host-side interface
extern "C" void quasirandomGeneratorGPU(double *d_Output, unsigned int seed, unsigned int N)
{
    dim3 threads(128, QRNG_DIMENSIONS);
    quasirandomGeneratorKernel<<<128, threads>>>(d_Output, seed, N);
    getLastCudaError("quasirandomGeneratorKernel() execution failed.\n");
}



////////////////////////////////////////////////////////////////////////////////
// Moro's Inverse Cumulative Normal Distribution function approximation
////////////////////////////////////////////////////////////////////////////////
__device__ inline double MoroInvCNDgpu(unsigned int x)
{
    const double a1 = 2.50662823884f;
    const double a2 = -18.61500062529f;
    const double a3 = 41.39119773534f;
    const double a4 = -25.44106049637f;
    const double b1 = -8.4735109309f;
    const double b2 = 23.08336743743f;
    const double b3 = -21.06224101826f;
    const double b4 = 3.13082909833f;
    const double c1 = 0.337475482272615f;
    const double c2 = 0.976169019091719f;
    const double c3 = 0.160797971491821f;
    const double c4 = 2.76438810333863E-02f;
    const double c5 = 3.8405729373609E-03f;
    const double c6 = 3.951896511919E-04f;
    const double c7 = 3.21767881768E-05f;
    const double c8 = 2.888167364E-07f;
    const double c9 = 3.960315187E-07f;

    double z;

    bool negate = false;

    // Ensure the conversion to doubleing point will give a value in the
    // range (0,0.5] by restricting the input to the bottom half of the
    // input domain. We will later reflect the result if the input was
    // originally in the top half of the input domain
    if (x >= 0x80000000UL)
    {
        x = 0xffffffffUL - x;
        negate = true;
    }

    // x is now in the range [0,0x80000000) (i.e. [0,0x7fffffff])
    // Convert to doubleing point in (0,0.5]
    const double x1 = 1.0f / static_cast<double>(0xffffffffUL);
    const double x2 = x1 / 2.0f;
    double p1 = x * x1 + x2;
    // Convert to doubleing point in (-0.5,0]
    double p2 = p1 - 0.5f;

    // The input to the Moro inversion is p2 which is in the range
    // (-0.5,0]. This means that our output will be the negative side
    // of the bell curve (which we will reflect if "negate" is true).

    // Main body of the bell curve for |p| < 0.42
    if (p2 > -0.42f)
    {
        z = p2 * p2;
        z = p2 * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0f);
    }
    // Special case (Chebychev) for tail
    else
    {
        z = __logf(-__logf(p1));
        z = - (c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9))))))));
    }

    // If the original input (x) was in the top half of the range, reflect
    // to get the positive side of the bell curve
    return negate ? -z : z;
}

////////////////////////////////////////////////////////////////////////////////
// Main kernel. Choose between transforming
// input sequence and uniform ascending (0, 1) sequence
////////////////////////////////////////////////////////////////////////////////
static __global__ void inverseCNDKernel(
    double *d_Output,
    unsigned int *d_Input,
    unsigned int pathN
)
{
    unsigned int distance = ((unsigned int)-1) / (pathN + 1);
    unsigned int     tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int threadN = MUL(blockDim.x, gridDim.x);

    //Transform input number sequence if it's supplied
    if (d_Input)
    {
        for (unsigned int pos = tid; pos < pathN; pos += threadN)
        {
            unsigned int d = d_Input[pos];
            d_Output[pos] = (double)MoroInvCNDgpu(d);
        }
    }
    //Else generate input uniformly placed samples on the fly
    //and write to destination
    else
    {
        for (unsigned int pos = tid; pos < pathN; pos += threadN)
        {
            unsigned int d = (pos + 1) * distance;
            d_Output[pos] = (double)MoroInvCNDgpu(d);
        }
    }
}

extern "C" void inverseCNDgpu(double *d_Output, unsigned int *d_Input, unsigned int N)
{
    inverseCNDKernel<<<128, 128>>>(d_Output, d_Input, N);
    getLastCudaError("inverseCNDKernel() execution failed.\n");
}



#endif