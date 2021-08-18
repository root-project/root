#ifndef RBC_H
#define RBC_H

#include "RooSpan.h"

#include "TError.h"

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

namespace RooBatchCompute {

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

struct RunContext;
// We have to use map instead of unordered_map because the unordered_maps from
// nvcc and gcc are not compatible sometimes.
typedef std::map<const RooAbsReal*,RooSpan<const double>> DataMap;
typedef std::vector<const RooAbsReal*> VarVector;
typedef std::vector<double> ArgVector;
typedef double* __restrict RestrictArr;
typedef const double* __restrict InputArr;

}
namespace rbc=RooBatchCompute;

#endif //#ifndef RBC_H
