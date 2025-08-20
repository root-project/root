#ifndef BtagVertex_v1_h
#define BtagVertex_v1_h

#include "DataVector.h"
#include <vector>
#include <utility>

namespace xAOD {
   struct BTagging_v1 : public SG::AuxElement
   {
      int fTagging;
   };

   struct BTagVertex_v1 : public SG::AuxElement
   {
      int fBtag;
   };

   typedef DataVector<BTagVertex_v1> BTagVertexContainer_v1;
   typedef DataVector<BTagging_v1> BTaggingContainer_v1;
}

#endif
