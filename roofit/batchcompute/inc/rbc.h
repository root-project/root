/**
\file rbc.h
\class RbcInterface
\ingroup Roobatchcompute

This file contains all the necessary declarations for using the RooBatchCompute library.
The class RbcInterface provides the mechanism for external modules (like RooFit) to call
functions from the library. The power lies in the virtual functions that can resolve to different
implementations for the functionality; for example, calling a function through dispatchCuda
will resolve to efficient cuda implementations.
**/ 
#ifndef RBC_H
#define RBC_H

#include "RooSpan.h"

#include "TError.h"
#include "DllImport.h" //for R__EXTERN, needed for windows

#include <functional>
#include <map>
#include <vector>

#ifndef __CUDACC__
  #define __device__
  #define __global__
  #define __host__
  struct cudaEvent_t;
  struct cudaStream_t;
#endif // #ifndef __CUDACC__

class RooAbsReal;

namespace rbc {

#ifdef __CUDACC__
  #define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
  inline void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
  {
    if (error != cudaSuccess)
      {
        Fatal( (func+"(), "+file+":"+std::to_string(line)).c_str(), "%s(%d)", cudaGetErrorString(error), error );
        throw std::bad_alloc();
      }
  }      
#endif

enum Computer{AddPdf, ArgusBG, Bernstein, BifurGauss, BreitWigner, Bukin, CBShape, Chebychev,
              ChiSquare, DstD0BG, Exponential, Gamma, Gaussian, Johnson, Landau, Lognormal,
              NegativeLogarithms, Novosibirsk, Poisson, Polynomial, ProdPdf, Voigtian};

enum BatchMode{Off, Cpu, Cuda};

struct RunContext;
// We have to use map instead of unordered_map because the unordered_maps from
// nvcc and gcc are not compatible sometimes.
typedef std::map<const RooAbsReal*,RooSpan<const double>> DataMap;
typedef std::vector<const RooAbsReal*> VarVector;
typedef std::vector<double> ArgVector;
typedef double* __restrict RestrictArr;
typedef const double* __restrict InputArr;

class RbcInterface {
  public:
    virtual ~RbcInterface() = default;
    virtual void   init() { throw std::bad_function_call(); }
    virtual void   compute(Computer, RestrictArr, size_t, const DataMap&, const VarVector&, const ArgVector& ={}) = 0;
    virtual double sumReduce(InputArr, size_t) = 0;
       
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
}; // end class RbcInterface

R__EXTERN RbcInterface *dispatchCPU, *dispatchCUDA;

} // end namespace rbc

#endif //#ifndef RBC_H
