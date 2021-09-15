// RooBatchCompute library created September 2020 by Emmanouil Michalainas
/**
\file RooBatchCompute.cu
\class RbcClass
\ingroup Roobatchcompute

This file contains the code for cuda computations using the RooBatchCompute library.
**/ 
#include "rbc.h"
#include "Batches.h"

#include "TError.h"

#include <thrust/reduce.h>

namespace rbc {
namespace RF_ARCH {
  
std::vector<void(*)(Batches)> getFunctions();
/// This class overrides some RbcInterface functions, for the purpose of providing
/// a cuda specific implementation of the library.
class RbcClass : public RbcInterface {
  private:
    const std::vector<void(*)(Batches)> _computeFunctions;
  public:
    RbcClass() : _computeFunctions(getFunctions())
    {
      dispatchCUDA = this; // Set the dispatch pointer to this instance of the library upon loading
    }

    /** Initialize the cuda computation library.
    This method needs to be called after the dynamic loading of the cuda instance of the
    RooBatchCompute library. If cuda is not working properly, it will set the dispatchCUDA
    pointer to nullptr. **/
    void init()
    {
      cudaError_t err = cudaSetDevice(0);
      if (err==cudaSuccess) cudaFree(nullptr);
      else
      {
        dispatchCUDA = nullptr;
        Error( (std::string(__func__)+"(), "+__FILE__+":"+std::to_string(__LINE__)).c_str(), "%s", cudaGetErrorString(err) );
      }
    }
    /** Compute multiple values using cuda kernels.
    This method creates a Batches object and passes it to the correct compute function.
    The compute function is launched as a cuda kernel.
    \param computer An enum specifying the compute function to be used.
    \param output The array where the computation results are stored.
    \param nEvents The number of events to be processed.
    \param varData A std::map containing the values of the variables involved in the computation.
    \param vars A std::vector containing pointers to the variables involved in the computation.
    \param extraArgs An optional std::vector containing extra double values that may participate in the computation. **/ 
    void compute(Computer computer, RestrictArr output, size_t nEvents, const DataMap& varData, const VarVector& vars, const ArgVector& extraArgs) override  
    {
      Batches batches(output, nEvents, varData, vars, extraArgs);
      _computeFunctions[computer]<<<128,512>>>(batches);
    }
    /// Return the sum of an input array
    double sumReduce(InputArr input, size_t n) override {
      return thrust::reduce(thrust::device, input, input+n, 0.0);
    }

    //cuda functions
    virtual void* cudaMalloc(size_t nBytes) {
      void* ret;
      ERRCHECK( ::cudaMalloc(&ret, nBytes) );
      return ret;
    }
    virtual void cudaFree(void* ptr) {
      ERRCHECK( ::cudaFree(ptr) );
    }
    virtual void* cudaMallocHost(size_t nBytes) {
      void* ret;
      ERRCHECK( ::cudaMallocHost(&ret, nBytes) );
      return ret;
    }
    virtual void cudaFreeHost(void* ptr) {
      ERRCHECK( ::cudaFreeHost(ptr) );
    }
    virtual cudaEvent_t* newCudaEvent(bool forTiming) {
      auto ret = new cudaEvent_t;
      ERRCHECK( cudaEventCreateWithFlags(ret, forTiming ? 0 : cudaEventDisableTiming) );
      return ret;
    }
    virtual void deleteCudaEvent(cudaEvent_t* event) {
      ERRCHECK( cudaEventDestroy(*event) );
      delete event;
    }
    virtual void cudaEventRecord(cudaEvent_t* event, cudaStream_t* stream) {
      ERRCHECK( ::cudaEventRecord(*event, *stream) );
    }
    virtual cudaStream_t* newCudaStream() {
      auto ret = new cudaStream_t;
      ERRCHECK( cudaStreamCreate(ret) );
      return ret;
    }
    virtual void deleteCudaStream(cudaStream_t* stream) {
      ERRCHECK( cudaStreamDestroy(*stream) );
      delete stream;
    }
    virtual bool streamIsActive(cudaStream_t* stream) {
      cudaError_t err = cudaStreamQuery(*stream);
      if (err==cudaErrorNotReady) return true;
      else if (err==cudaSuccess) return false;
      ERRCHECK(err);
      return false;
    }
    virtual void cudaStreamWaitEvent(cudaStream_t* stream, cudaEvent_t* event) {
      ERRCHECK( ::cudaStreamWaitEvent(*stream, *event) );
    }
    virtual float cudaEventElapsedTime(cudaEvent_t* begin, cudaEvent_t* end) {
      float ret;
      ERRCHECK( ::cudaEventElapsedTime(&ret, *begin, *end) );
      return ret;
    }
    void memcpyToCUDA(void* dest, const void* src, size_t nBytes, cudaStream_t* stream) override {
      if (stream)
        ERRCHECK( cudaMemcpyAsync(dest, src, nBytes, cudaMemcpyHostToDevice, *stream) );
      else
        ERRCHECK( cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice) );
    }
    void memcpyToCPU(void* dest, const void* src, size_t nBytes, cudaStream_t* stream) override {
      if (stream)
        ERRCHECK( cudaMemcpyAsync(dest, src, nBytes, cudaMemcpyDeviceToHost, *stream) );
      else
        ERRCHECK( cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost) );
    }
}; // End class RbcClass

/// Static object to trigger the constructor which overwrites the dispatch pointer.
static RbcClass computeObj;

/** Construct a Batches object
\param output The array where the computation results are stored.
\param nEvents The number of events to be processed.
\param varData A std::map containing the values of the variables involved in the computation.
\param vars A std::vector containing pointers to the variables involved in the computation.
\param extraArgs An optional std::vector containing extra double values that may participate in the computation.
For every scalar parameter a `Batch` object inside the `Batches` object is set accordingly;
a data member of type double gets assigned the scalar value. This way, when the cuda kernel
is launched this scalar value gets copied automatically and thus no call to cudaMemcpy is needed **/ 
Batches::Batches(RestrictArr output, size_t nEvents, const DataMap& varData, const VarVector& vars, const ArgVector& extraArgs, double[maxParams][bufferSize])
  : _nEvents(nEvents), _nBatches(vars.size()), _nExtraArgs(extraArgs.size()), _output(output)
{  
  for (int i=0; i<vars.size(); i++)
  {
    const RooSpan<const double>& span = varData.at(vars[i]);
    size_t size = span.size();
    if (size==1) _arrays[i].set(span[0], nullptr, false);
    else _arrays[i].set(0.0, span.data(), true);
  }
  std::copy(extraArgs.cbegin(), extraArgs.cend(), _extraArgs);
}

} // End namespace RF_ARCH
} // End namespace rbc
