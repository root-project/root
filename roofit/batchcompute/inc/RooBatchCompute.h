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

#ifndef ROOFIT_BATCHCOMPUTE_ROOBATCHCOMPUTE_H
#define ROOFIT_BATCHCOMPUTE_ROOBATCHCOMPUTE_H

#include "RooBatchComputeTypes.h"

#include "DllImport.h" //for R__EXTERN, needed for windows
#include "TError.h"

#include <functional>

/**
 * Namespace for dispatching RooFit computations to various backends.
 *
 * This namespace contains an interface for providing high-performance computation functions for use in RooAbsReal::evaluateSpan(),
 * see RooBatchComputeInterface.
 *
 * Furthermore, several implementations of this interface can be created, which reside in RooBatchCompute::RF_ARCH, where
 * RF_ARCH may be replaced by the architecture that this implementation targets, e.g. SSE, AVX, etc.
 *
 * Using the pointer RooBatchCompute::dispatch, a computation request can be dispatched to the fastest backend that is available
 * on a specific platform.
 */
namespace RooBatchCompute {

enum class Architecture { AVX512, AVX2, AVX, SSE4, GENERIC, CUDA };

enum Computer{AddPdf, ArgusBG, Bernstein, BifurGauss, BreitWigner, Bukin, CBShape, Chebychev,
              ChiSquare, DstD0BG, Exponential, Gamma, Gaussian, Johnson, Landau, Lognormal,
              NegativeLogarithms, Novosibirsk, Poisson, Polynomial, ProdPdf, Voigtian};

/**
 * \class RooBatchComputeInterface
 * \ingroup Roobatchcompute
 * \brief The interface which should be implemented to provide optimised computation functions for implementations of RooAbsReal::evaluateSpan().
 *
 * The class RooBatchComputeInterface provides the mechanism for external modules (like RooFit) to call
 * functions from the library. The power lies in the virtual functions that can resolve to different
 * implementations for the functionality; for example, calling a function through dispatchCuda
 * will resolve to efficient cuda implementations.
 *
 * This interface contains the signatures of the compute functions of every PDF that has an optimised implementation available.
 * These are the functions that perform the actual computations in batches.
 *
 * Several implementations of this interface may be provided, e.g. SSE, AVX, AVX2 etc. At run time, the fastest implementation of this interface
 * is selected, and using a virtual call, the computation is dispatched to the best backend.
 *
 * \see RooBatchCompute::dispatch, RooBatchComputeClass, RF_ARCH
 */ 
class RooBatchComputeInterface {
  public:
    virtual ~RooBatchComputeInterface() = default;
    virtual void   compute(cudaStream_t*, Computer, RestrictArr, size_t, const DataMap&, const VarVector&, const ArgVector& ={}) = 0;
    virtual double sumReduce(cudaStream_t*, InputArr input, size_t n) = 0;
    virtual Architecture architecture() const = 0;
    virtual std::string architectureName() const = 0;
       
    //cuda functions that need to be interfaced
    virtual void* cudaMalloc(size_t)                { throw std::bad_function_call(); }
    virtual void  cudaFree(void*)                   { throw std::bad_function_call(); }
    virtual void* cudaMallocHost(size_t)            { throw std::bad_function_call(); }
    virtual void  cudaFreeHost(void*)               { throw std::bad_function_call(); }
    virtual cudaEvent_t* newCudaEvent(bool /*forTiming*/) { throw std::bad_function_call(); }
    virtual void  deleteCudaEvent(cudaEvent_t*)     { throw std::bad_function_call(); }
    virtual cudaStream_t* newCudaStream()           { throw std::bad_function_call(); }
    virtual void  deleteCudaStream(cudaStream_t*)   { throw std::bad_function_call(); }
    virtual bool  streamIsActive(cudaStream_t*)     { throw std::bad_function_call(); }
    virtual void  cudaEventRecord(cudaEvent_t*, cudaStream_t*)     { throw std::bad_function_call(); }
    virtual void  cudaStreamWaitEvent(cudaStream_t*, cudaEvent_t*) { throw std::bad_function_call(); }
    virtual float cudaEventElapsedTime(cudaEvent_t*, cudaEvent_t*) { throw std::bad_function_call(); }
    virtual void  memcpyToCUDA(void*, const void*, size_t, cudaStream_t* =nullptr) { throw std::bad_function_call(); }
    virtual void  memcpyToCPU (void*, const void*, size_t, cudaStream_t* =nullptr) { throw std::bad_function_call(); }
};

/**
 * This dispatch pointer points to an implementation of the compute library, provided one has been loaded.
 * Using a virtual call, computation requests are dispatched to backends with architecture-specific functions
 * such as SSE, AVX, AVX2, etc.
 *
 * \see RooBatchComputeInterface, RooBatchComputeClass, RF_ARCH
 */
R__EXTERN RooBatchComputeInterface *dispatchCPU, *dispatchCUDA;
} // End namespace RooBatchCompute

#endif
