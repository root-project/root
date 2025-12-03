#ifndef Jet_v1_h
#define Jet_v1_h

#include "DataVector.h"
#include <vector>
#include <utility>

namespace xAOD {
   struct Jet_v1;
}

template <> struct DataVectorBase<xAOD::Jet_v1>
{
   typedef DataVector<xAOD::IParticle > Base;
   int fJet;
};

namespace xAOD {
   struct Jet_v1 : public xAOD::IParticle
   {
      int fJet;
   };

   typedef DataVector<Jet_v1> JetContainer_v1;
}

#endif
