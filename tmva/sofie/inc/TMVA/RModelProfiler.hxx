#ifndef TMVA_SOFIE_RMODELPROFILER
#define TMVA_SOFIE_RMODELPROFILER

#include "TMVA/RModel.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModelProfiler {
private:
   RModel &fModel;
   
   void GenerateUtilityFunctions();

public:
   // The profiler must be constructed with a model to work on.
   RModelProfiler() = delete;
   RModelProfiler(RModel &model);
   ~RModelProfiler() = default;
   
   // There is no point in copying or moving an RModelProfiler
   RModelProfiler(const RModelProfiler &other) = delete;
   RModelProfiler(RModelProfiler &&other) = delete;
   RModelProfiler &operator=(const RModelProfiler &other) = delete;
   RModelProfiler &operator=(RModelProfiler &&other) = delete;
   
   // Main function to generate the profiled code.
   void Generate();
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_RMODELPROFILER