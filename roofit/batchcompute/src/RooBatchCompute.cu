// rbc library created September 2020 by Emmanouil Michalainas

#include "rbc.h"
#include "Batches.h"

#include "TError.h"

#include <thrust/reduce.h>

namespace rbc {
namespace RF_ARCH {
  
std::vector<void(*)(Batches)> getFunctions();

class RbcClass : public RbcInterface {
  private:
    const std::vector<void(*)(Batches)> _computeFunctions;
  public:
    RbcClass() : _computeFunctions(getFunctions())
    {
      dispatchCUDA = this; // Set the dispatch pointer to this instance of the library upon loading
    }

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

    void compute(Computer computer, RestrictArr output, size_t nEvents, const DataMap& varData, const VarVector& vars, const ArgVector& extraArgs) override  
    {
      Batches batches(output, nEvents, varData, vars, extraArgs);
      _computeFunctions[computer]<<<128,512>>>(batches);
    }
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
