#ifndef EXAMPLE_MC_H_
#define EXAMPLE_MC_H_

#include <Rtypes.h>

struct EvolutionStruct_V2 {
   float fOldMember = 0.;

   ClassDefNV(EvolutionStruct_V2, 2);
};

struct EvolutionStruct_V3 {
   float fNewMember = 0.;

   ClassDefNV(EvolutionStruct_V3, 3);
};

#endif
