#ifndef EXAMPLE_MC_H_
#define EXAMPLE_MC_H_

#include <Rtypes.h>
#include <TClonesArray.h>
#include <TObject.h>

struct EvolutionStruct_V2 {
   float fOldMember = 0.;

   ClassDefNV(EvolutionStruct_V2, 2);
};

struct EvolutionStruct_V3 {
   float fNewMember = 0.;

   ClassDefNV(EvolutionStruct_V3, 3);
};

struct EvolutionClonesItem_V2 : public TObject {
   float fPx = 0.;
   float fPy = 0.;
   float fPz = 0.;
   EvolutionClonesItem_V2() = default;
   EvolutionClonesItem_V2(float px, float py, float pz) : fPx(px), fPy(py), fPz(pz) {}

   ClassDefOverride(EvolutionClonesItem_V2, 2);
};

struct EvolutionClonesItem_V3 : public TObject {
   int fPz = 0;
   int fCanary1 = 33; ///<! Marked transient must be unchanged.
   int fCanary2 = 66; ///<! Marked transient must be unchanged.
   int fCombined = 0;

   ClassDefOverride(EvolutionClonesItem_V3, 3);
};

struct EvolutionClonesHolder_V2 : public TObject {
   TClonesArray *fItems = nullptr;
   EvolutionClonesHolder_V2() { fItems = new TClonesArray("EvolutionClonesItem_V2", 10); }
   ~EvolutionClonesHolder_V2() override { delete fItems; }

   ClassDefOverride(EvolutionClonesHolder_V2, 2);
};

struct EvolutionClonesHolder_V3 : public TObject {
   TClonesArray *fItems = nullptr;
   EvolutionClonesHolder_V3() { fItems = new TClonesArray("EvolutionClonesItem_V3", 10); }
   ~EvolutionClonesHolder_V3() override { delete fItems; }

   ClassDefOverride(EvolutionClonesHolder_V3, 3);
};

#endif
