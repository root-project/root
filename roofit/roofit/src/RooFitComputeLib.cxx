/* RooFitComputeLib.cxx - created September 2020
 */
#include "RooFitComputeLib.h"
#include "RooVDTHeaders.h"
#include "BatchHelpers.h"
#include "RunContext.h"

using namespace BatchHelpers;

namespace RooFitCompute {

  /// Small helping function that determines the sizes of the batches and decides
  /// wether to run the High performing or the Generic computation function.
  struct AnalysisInfo {
    size_t batchSize=SIZE_MAX;
    bool canDoHighPerf=true;
  };
  AnalysisInfo analyseInputSpans(std::vector<RooSpan<const double>> parameters)
  {
    AnalysisInfo ret;
    if (parameters[0].size()<=1) ret.canDoHighPerf=false;
    else ret.batchSize = std::min(ret.batchSize, parameters[0].size());
    for (size_t i=1; i<parameters.size(); i++)
      if (parameters[i].size()>1)
      {
        ret.canDoHighPerf=false;
        ret.batchSize = std::min(ret.batchSize, parameters[i].size());
      }
    return ret;
  }

  /// Templated function that works for every PDF: does the necessary preprocessing and launches
  /// the correct overload of the actual computing function. 
  template <class Computer_t, typename Arg_t, typename... Args_t>
  RooSpan<double> startComputation(const RooAbsReal* caller, RunContext& evalData, Computer_t computer, Arg_t first, Args_t... rest)
  {
    AnalysisInfo info = analyseInputSpans({first, rest...});
    RooSpan<double> output = evalData.makeBatch(caller, info.batchSize);
    
    if (info.canDoHighPerf) computer.run(info.batchSize, output.data(), first, BracketAdapter<double>(rest[0])...);
    else                    computer.run(info.batchSize, output.data(), BracketAdapterWithMask(first), BracketAdapterWithMask(rest)...);

    return output;
  }

  /* Special namespace with a macro as it's identifier. Used for differienting the names of the identical
   * functions that should be distinct, because they are compiled for different vector architecture,
   * SSE4.1, AVX2 and AVX512f                                                                               */
  namespace RF_ARCH {
    //create a dummy object of _RooFitCompute class, so that the constructor gets triggered and the dispatch pointer is overwritten
    static RooFitComputeClass computeObj;

    //constructor of _RooFitCompute 
    RooFitComputeClass::RooFitComputeClass() {
      RooFitCompute::dispatch = this;
    }


  } //End namespace RF_ARCH
} //End namespace RooFitCompute
