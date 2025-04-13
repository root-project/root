namespace LHCb {

   class Particle {};

} // namespace Particle

namespace LoKi {

   template< typename T > class Constant {};
   template< typename T > class BooleanConstant{};

} // namespace Loki

#ifdef __CINT__
#pragma link C++ class LoKi::Constant< const LHCb::Particle* >;
#pragma link C++ class LoKi::BooleanConstant< const LHCb::Particle* >;
#endif
