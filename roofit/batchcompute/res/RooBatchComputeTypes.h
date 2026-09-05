/*
 * Project: RooFit
 * Authors:
 *   Emmanouil Michalainas, CERN 6 January 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOFIT_BATCHCOMPUTE_ROOBATCHCOMPUTETYPES_H
#define ROOFIT_BATCHCOMPUTE_ROOBATCHCOMPUTETYPES_H


#ifdef __CUDACC__
#define __roodevice__ __device__
#define __roohost__ __host__
#define __rooglobal__ __global__
#else
#define __roodevice__
#define __roohost__
#define __rooglobal__
#endif // #indef __CUDACC__

// Double-precision atomicAdd() is only provided by the CUDA runtime for
// compute capability 6.0 and higher. This is the canonical fallback
// implementation from the CUDA C++ Programming Guide for older devices.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double *address, double val)
{
   unsigned long long int *address_as_ull = (unsigned long long int *)address;
   unsigned long long int old = *address_as_ull;
   unsigned long long int assumed;
   do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
   } while (assumed != old);
   return __longlong_as_double(old);
}
#endif

#endif
