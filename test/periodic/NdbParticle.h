#ifndef PARTICLE_H
#define PARTICLE_H

#include <TObject.h>
#include <TString.h>

/* ------------- Particle Types ---------------- */
enum   ParticleType {
   PTGamma,
   PTNeutron,
   PTAntiNeutron,
   PTNeutrino,
   PTAntiNeutrino,
   PTProton,
   PTAntiProton,
   PTElectron,
   PTPositron,
   PTPionPlus,
   PTPionMinus,
   PTMuonPlus,
   PTMuonMinus,
   PTKaonPlus,
   PTKaonMinus,
   PTKaonLong,
   PTKaonShort,
   PTKaonZero,
   PTLambda,
   PTAlpha
};

/* ------------ Particle Class ---------------- */
class NdbParticle : public TObject
{
protected:
   Long_t       pId;      // Unique id of each particle
   TString      pName;      // Particle Name
   TString      pMnemonic;   // Particle Mnemonic
   ParticleType pType;      // Type of particle
   Int_t        pCharge;   // Particle charge
   Float_t      pMass;      // mass in MeV/c^2
   Float_t      pHalfLife;   // Half life in sec.

public:
   NdbParticle() { pId = -1; }

   NdbParticle(
      const char *aName,
      ParticleType aType,
      long anId = 0)
   : TObject(), pName(aName)
   {
      pType = aType;
      pId = anId;
   }

   ~NdbParticle() {}

   // --- Access Functions ---
   inline TString Name()      const { return pName; }
   inline TString Mnemonic()   const { return pMnemonic; }
   inline Long_t  Id()      const { return pId; }
   inline Float_t Mass()      const { return pMass; }
   inline Float_t HalfLife()   const { return pHalfLife; }
   inline Int_t   Charge()      const { return pCharge; }

   ClassDef(NdbParticle,1)

}; // NdbParticle

#endif
