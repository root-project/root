#ifndef REACTIONXS_H
#define REACTIONXS_H

#include <TArrayF.h>

#include "NdbDefs.h"
#include "NdbMF.h"
#include "NdbMT.h"
#include "NdbParticleList.h"
#include "NdbMaterial.h"

// --- Interpolation types ----
#define   IT_HISTOGRAM   1
#define   IT_LINLIN   2
#define   IT_LINLOG   3
#define   IT_LOGLIN   4
#define   IT_LOGLOG   5
#define   IT_GAMOW   6

/* ========= NdbMTReactionXS ============ */
class NdbMTReactionXS : public NdbMT
{
protected:
   TArrayF      ene;         // Energy in eV
   TArrayF      xs;         // Cross section in barn

   Float_t      minxs,         // Minimum and
         maxxs;         // Maximum limits of XS

   NdbParticle   projectile;      // Projectile particle
   NdbParticleList   daugthers;      // Reaction products
   NdbMaterial   residual;      // Residual nucleus
   Double_t   QM;         // Mass-difference Q value (eV)
   Double_t   QI;         // reaction Q for the lowest
                  // energy state
   Int_t      LR;         // Complex or "breakup" flag.
   Int_t      NP;         // No. Points (x,y)
   Int_t      NR;         // interpolation regions
   Int_t      IT;         // Interpolation type

public:
   NdbMTReactionXS(int aMt, const char *desc)
   : NdbMT(aMt,desc) {
      LR = -1;
      NP = -1;
      NR = -1;
      minxs = maxxs = 0.0;
   }
   ~NdbMTReactionXS() {}

   // --- Access functions ---
   inline Float_t   Energy(int i)      { return ene[i]; }
   inline Float_t   XS(int i)      { return xs[i]; }

   inline Int_t   Pairs()         const   { return NP; }
   inline Int_t   InterpolationType()   const   { return IT; }
   inline Int_t   InterpolationRegions()   const   { return NR; }
   inline Float_t   MassDifference()   const   { return QM; }
   inline Float_t   ReactionQ()      const   { return QI; }

   // --- Input/Output routines ---
   Bool_t      LoadENDF(char *filename);

   // --- Interpolation routines ---
   Int_t      BinSearch( Float_t e);
   Float_t      Interpolate(Float_t e);

   // --- Limits ---
   inline Float_t   MinEnergy()         { return ene[0]; }
   inline Float_t   MaxEnergy()         { return ene[NP-1]; }
   inline Float_t   MinXS()         const   { return minxs; }
   inline Float_t   MaxXS()         const   { return maxxs; }

   ClassDef(NdbMTReactionXS,1)

}; // NdbMTReactionXS

#endif
