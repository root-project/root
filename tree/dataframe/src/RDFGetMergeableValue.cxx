#include <string>
#include <vector>

#include "ROOT/RResultPtr.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "ROOT/RDF/RLoopManager.hxx"

namespace ROOT {
namespace Detail {
namespace RDF {

////////////////////////////////////////////////////////////////////////////////
/// \brief Retrieve a mergeable value from an RDataFrame Snapshot action.
/// \param[in] rptr lvalue reference of an RResultPtr of a Snapshot action.
/// \returns An RMergeableValue holding a vector with the path to the
///          snapshotted file.
///
/// This is a specialized overload of GetMergeableValue for the Snapshot action.
/// Usually, the type of the result held by the RResultPtr is the same as the
/// type of the value stored in the RMergeableValue. For this operation, we
/// don't store the dataframe held by the RResultPtr, but just the path to the
/// file with the snapshotted data. In the merge phase, paths of different files
/// are concatenated in a vector, so that the user can build a TChain with it.
std::unique_ptr<RMergeableValue<std::vector<std::string>>>
GetMergeableValue(RResultPtr<ROOT::RDF::RInterface<RLoopManager, void>> &rptr)
{
   rptr.ThrowIfNull();
   if (!rptr.fActionPtr->HasRun())
      rptr.TriggerRun(); // Prevents from using `const` specifier in parameter
   return std::unique_ptr<RMergeableValue<std::vector<std::string>>>{
      static_cast<RMergeableValue<std::vector<std::string>> *>(rptr.fActionPtr->GetMergeableValue().release())};
}

} // namespace RDF
} // namespace Detail
} // namespace ROOT
