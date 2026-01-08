#ifndef TMVA_SOFIE_RMODELPROFILER
#define TMVA_SOFIE_RMODELPROFILER

#include "TMVA/RModel.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

/// \class RModelProfiler
/// \brief A helper class to generate profiled inference code for an RModel.
///
/// This class instruments the generated C++ code to measure the execution
/// time of each operator. The functions are invoked when the RModel::Generate is called
/// with the Options::kProfile flag.
class RModelProfiler {

public:
   static void AddNeededStdLibs(RModel &model);
   static std::string GenerateUtilityFunctions();
   static std::string GenerateSessionMembers();
   static std::string GenerateBeginInferCode();
   static std::string GenerateOperatorCode(ROperator &op, size_t op_idx);
   static std::string GenerateEndInferCode();


   RModelProfiler() = delete;
   ~RModelProfiler() = default;

   // There is no point in copying or moving an RModelProfiler
   RModelProfiler(const RModelProfiler &other) = delete;
   RModelProfiler(RModelProfiler &&other) = delete;
   RModelProfiler &operator=(const RModelProfiler &other) = delete;
   RModelProfiler &operator=(RModelProfiler &&other) = delete;

};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_RMODELPROFILER
