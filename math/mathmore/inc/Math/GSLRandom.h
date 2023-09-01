#ifndef ROOT_Math_GSLRandom
#define ROOT_Math_GSLRandom

#include "Math/Random.h"
#include "Math/GSLRndmEngines.h"

namespace ROOT {
namespace Math {


   typedef   Random<ROOT::Math::GSLRngMT>     RandomMT;
   typedef   Random<ROOT::Math::GSLRngTaus>   RandomTaus;
   typedef   Random<ROOT::Math::GSLRngRanLux> RandomRanLux;
   typedef   Random<ROOT::Math::GSLRngGFSR4>  RandomGFSR4;


} // namespace Math
} // namespace ROOT

#endif
