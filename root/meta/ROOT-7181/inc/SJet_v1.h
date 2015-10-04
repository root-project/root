#ifndef SJet_v1_h
#define SJet_v1_h

#include "DataVector.h"
#include <vector>
#include <utility>

namespace xAOD {
   struct SJet_v1;
}

template <> struct DataVectorBase<xAOD::SJet_v1>
{
   typedef DataVector<xAOD::IParticle > Base;
   int fSJet;
};

namespace xAOD {
   struct SJet_v1 : public xAOD::IParticle
   {
      int fSJet;
   };

   typedef SDataVector<SJet_v1> SJetContainer_v1;
}

#endif
