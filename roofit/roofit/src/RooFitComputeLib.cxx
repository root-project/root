// RooFitCompute Library created September 2020 by Emmanouil Michalainas
#include "RooFitComputeInterface.h"

#include "RooVDTHeaders.h"
#include "BatchHelpers.h"
#include "RunContext.h"

using namespace BatchHelpers;

namespace RooFitCompute {
  namespace RF_ARCH {

    class RooFitComputeClass : public RooFitComputeInterface {
      private:
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

        template <class Computer_t, typename Arg_t, typename... Args_t>
        RooSpan<double> startComputation(const RooAbsReal* caller, RunContext& evalData, Computer_t computer, Arg_t first, Args_t... rest)
        {
          AnalysisInfo info = analyseInputSpans({first, rest...});
          RooSpan<double> output = evalData.makeBatch(caller, info.batchSize);

          if (info.canDoHighPerf) computer.run(info.batchSize, output.data(), first, BracketAdapter<double>(rest[0])...);
          else                    computer.run(info.batchSize, output.data(), BracketAdapterWithMask(first), BracketAdapterWithMask(rest)...);

          return output;
        }

      public:
        RooFitComputeClass() {
          RooFitCompute::dispatch = this;
        }
    }; // End class RooFitComputeClass

    static RooFitComputeClass computeObj;

  } //End namespace RF_ARCH
} //End namespace RooFitCompute
