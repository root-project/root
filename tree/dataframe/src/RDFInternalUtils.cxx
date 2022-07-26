#include "ROOT/RDF/InternalUtils.hxx"

namespace ROOT {
namespace Internal {
namespace RDF {

ROOT::RDataFrame MakeDataFrameFromSpec(const RDatasetSpec &spec)
{
   return ROOT::RDataFrame(std::move(spec));
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
