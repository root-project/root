#ifndef TMVA_RBDT
#define TMVA_RBDT

#include "TMVA/RTensor.hxx"

#include <sstream> // std::stringstream

namespace TMVA {
namespace Experimental {

/// TMVA::Reader legacy interface
class RBDT {
private:

public:
   /// Construct from a weight file
   RBDT(const std::string& modelname, const std::string& filename) {
      // TODO: Read from the file <filename> the model <modelname> using the backend
   }

   /// Compute model prediction on vector
   std::vector<float> Compute(const std::vector<float> &x)
   {
      // TODO: Add inference for a single event
      return {0.0};
   }

   /// Compute model prediction on input RTensor
   RTensor<float> Compute(RTensor<float> &x)
   {
      // TODO: Add inference for a batch of events
      return RTensor<float>({x.GetShape()[0], 1});
   }
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RREADER
