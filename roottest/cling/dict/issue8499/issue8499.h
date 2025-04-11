#include <TClass.h>
#include <array>

namespace o2{
namespace track{
struct TrackParCov {
   double fParams[5];
};
}
template <int N, class C>
class DCAFitterN {
   std::array<C*, N> fMember;
};
namespace vertexing{
using DCAFitter2 = DCAFitterN<2, o2::track::TrackParCov>;
}
}
