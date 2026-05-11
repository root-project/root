#include <type_traits>

namespace LHCb {

class Particle {};

} // namespace LHCb

namespace LoKi {

template <typename T>
class Constant {
public:
   static constexpr bool isConstParticlePtr() { return std::is_same_v<T, LHCb::Particle const *>; }
};
template <typename T>
class BooleanConstant {
public:
   static constexpr bool isConstParticlePtr() { return std::is_same_v<T, LHCb::Particle const *>; }
};

} // namespace LoKi

#ifdef __CLING__
#pragma link C++ class LoKi::Constant< const LHCb::Particle* >;
#pragma link C++ class LoKi::BooleanConstant< const LHCb::Particle* >;
#endif
