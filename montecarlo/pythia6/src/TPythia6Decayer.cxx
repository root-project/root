// @(#)root/pythia6:$Id$
// Author: Christian Holm Christensen   22/04/06
// Much of this code has been lifted from AliROOT.

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// TPythia6Decayer                                                           //
//                                                                           //
// This implements the TVirtualMCDecayer interface.  The TPythia6            //
// singleton object is used to decay particles.  Note, that since this       //
// class modifies common blocks (global variables) it is defined as a        //
// singleton.                                                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TPythia6.h"
#include "TPythia6Decayer.h"
#include "TPDGCode.h"
#include "TLorentzVector.h"
#include "TClonesArray.h"

ClassImp(TPythia6Decayer)

TPythia6Decayer* TPythia6Decayer::fgInstance = 0;

//______________________________________________________________________________
TPythia6Decayer* TPythia6Decayer::Instance()
{
   // Get the singleton object.
   if (!fgInstance) fgInstance = new TPythia6Decayer;
   return fgInstance;
}

//______________________________________________________________________________
TPythia6Decayer::TPythia6Decayer()
   : fBraPart(501)
     fDecay(kMaxDecay)
{
   // Constructor
   fBraPart.Reset(1);
}

//______________________________________________________________________________
void TPythia6Decayer::Init()
{
   // Initialize the decayer
   static Bool_t init = kFALSE;
   if (init) return;
   init = kTRUE;
   ForceDecay();
}

//______________________________________________________________________________
void TPythia6Decayer::Decay(Int_t idpart, TLorentzVector* p)
{
   // Decay a particle of type IDPART (PDG code) and momentum P.
   if (!p) return;
   TPythia6::Instance()->Py1ent(0, idpart, p->Energy(), p->Theta(), p->Phi());
   TPythia6::Instance()->GetPrimaries();
}

//______________________________________________________________________________
Int_t TPythia6Decayer::ImportParticles(TClonesArray *particles)
{
   // Get the decay products into the passed PARTICLES TClonesArray of
   // TParticles
   return TPythia6::Instance()->ImportParticles(particles,"All");
}

//______________________________________________________________________________
void TPythia6Decayer::SetForceDecay(Int_t type)
{
   // Force a particular decay type
   if (type > kMaxDecay) {
      Warning("SetForceDecay", "Invalid decay mode: %d", type);
      return;
   }
   fDecay = EDecayType(type);
}

//______________________________________________________________________________
void TPythia6Decayer::ForceDecay()
{
   // Force a particle decay mode
   EDecayType decay=fDecay;
   TPythia6::Instance()->SetMSTJ(21,2);
   if (decay == kNoDecayHeavy) return;

   //
   // select mode
   Int_t products[3];
   Int_t mult[3];

   switch (decay) {
   case kHardMuons:
      products[0] =     13;
      products[1] =    443;
      products[2] = 100443;
      mult[0] = 1;
      mult[1] = 1;
      mult[2] = 1;
      ForceParticleDecay(  511, products, mult, 3);
      ForceParticleDecay(  521, products, mult, 3);
      ForceParticleDecay(  531, products, mult, 3);
      ForceParticleDecay( 5122, products, mult, 3);
      ForceParticleDecay( 5132, products, mult, 3);
      ForceParticleDecay( 5232, products, mult, 3);
      ForceParticleDecay( 5332, products, mult, 3);
      ForceParticleDecay( 100443, 443, 1);  // Psi'  -> J/Psi X
      ForceParticleDecay(    443,  13, 2);  // J/Psi -> mu+ mu-

      ForceParticleDecay(  411,13,1); // D+/-
      ForceParticleDecay(  421,13,1); // D0
      ForceParticleDecay(  431,13,1); // D_s
      ForceParticleDecay( 4122,13,1); // Lambda_c
      ForceParticleDecay( 4132,13,1); // Xsi_c
      ForceParticleDecay( 4232,13,1); // Sigma_c
      ForceParticleDecay( 4332,13,1); // Omega_c
      break;
   case kSemiMuonic:
      ForceParticleDecay(  411,13,1); // D+/-
      ForceParticleDecay(  421,13,1); // D0
      ForceParticleDecay(  431,13,1); // D_s
      ForceParticleDecay( 4122,13,1); // Lambda_c
      ForceParticleDecay( 4132,13,1); // Xsi_c
      ForceParticleDecay( 4232,13,1); // Sigma_c
      ForceParticleDecay( 4332,13,1); // Omega_c
      ForceParticleDecay(  511,13,1); // B0
      ForceParticleDecay(  521,13,1); // B+/-
      ForceParticleDecay(  531,13,1); // B_s
      ForceParticleDecay( 5122,13,1); // Lambda_b
      ForceParticleDecay( 5132,13,1); // Xsi_b
      ForceParticleDecay( 5232,13,1); // Sigma_b
      ForceParticleDecay( 5332,13,1); // Omega_b
      break;
   case kDiMuon:
      ForceParticleDecay(  113,13,2); // rho
      ForceParticleDecay(  221,13,2); // eta
      ForceParticleDecay(  223,13,2); // omega
      ForceParticleDecay(  333,13,2); // phi
      ForceParticleDecay(  443,13,2); // J/Psi
      ForceParticleDecay(100443,13,2);// Psi'
      ForceParticleDecay(  553,13,2); // Upsilon
      ForceParticleDecay(100553,13,2);// Upsilon'
      ForceParticleDecay(200553,13,2);// Upsilon''
      break;
   case kSemiElectronic:
      ForceParticleDecay(  411,11,1); // D+/-
      ForceParticleDecay(  421,11,1); // D0
      ForceParticleDecay(  431,11,1); // D_s
      ForceParticleDecay( 4122,11,1); // Lambda_c
      ForceParticleDecay( 4132,11,1); // Xsi_c
      ForceParticleDecay( 4232,11,1); // Sigma_c
      ForceParticleDecay( 4332,11,1); // Omega_c
      ForceParticleDecay(  511,11,1); // B0
      ForceParticleDecay(  521,11,1); // B+/-
      ForceParticleDecay(  531,11,1); // B_s
      ForceParticleDecay( 5122,11,1); // Lambda_b
      ForceParticleDecay( 5132,11,1); // Xsi_b
      ForceParticleDecay( 5232,11,1); // Sigma_b
      ForceParticleDecay( 5332,11,1); // Omega_b
      break;
   case kDiElectron:
      ForceParticleDecay(  113,11,2); // rho
      ForceParticleDecay(  333,11,2); // phi
      ForceParticleDecay(  221,11,2); // eta
      ForceParticleDecay(  223,11,2); // omega
      ForceParticleDecay(  443,11,2); // J/Psi
      ForceParticleDecay(100443,11,2);// Psi'
      ForceParticleDecay(  553,11,2); // Upsilon
      ForceParticleDecay(100553,11,2);// Upsilon'
      ForceParticleDecay(200553,11,2);// Upsilon''
      break;
   case kBJpsiDiMuon:

      products[0] =    443;
      products[1] = 100443;
      mult[0] = 1;
      mult[1] = 1;

      ForceParticleDecay(  511, products, mult, 2); // B0   -> J/Psi (Psi') X
      ForceParticleDecay(  521, products, mult, 2); // B+/- -> J/Psi (Psi') X
      ForceParticleDecay(  531, products, mult, 2); // B_s  -> J/Psi (Psi') X
      ForceParticleDecay( 5122, products, mult, 2); // Lambda_b -> J/Psi (Psi') X
      ForceParticleDecay( 100443, 443, 1);          // Psi'  -> J/Psi X
      ForceParticleDecay(    443,13,2);             // J/Psi -> mu+ mu-
      break;
   case kBPsiPrimeDiMuon:
      ForceParticleDecay(  511,100443,1); // B0
      ForceParticleDecay(  521,100443,1); // B+/-
      ForceParticleDecay(  531,100443,1); // B_s
      ForceParticleDecay( 5122,100443,1); // Lambda_b
      ForceParticleDecay(100443,13,2);    // Psi'
      break;
   case kBJpsiDiElectron:
      ForceParticleDecay(  511,443,1); // B0
      ForceParticleDecay(  521,443,1); // B+/-
      ForceParticleDecay(  531,443,1); // B_s
      ForceParticleDecay( 5122,443,1); // Lambda_b
      ForceParticleDecay(  443,11,2);  // J/Psi
      break;
   case kBJpsi:
      ForceParticleDecay(  511,443,1); // B0
      ForceParticleDecay(  521,443,1); // B+/-
      ForceParticleDecay(  531,443,1); // B_s
      ForceParticleDecay( 5122,443,1); // Lambda_b
      break;
   case kBPsiPrimeDiElectron:
      ForceParticleDecay(  511,100443,1); // B0
      ForceParticleDecay(  521,100443,1); // B+/-
      ForceParticleDecay(  531,100443,1); // B_s
      ForceParticleDecay( 5122,100443,1); // Lambda_b
      ForceParticleDecay(100443,11,2);   // Psi'
      break;
   case kPiToMu:
      ForceParticleDecay(211,13,1); // pi->mu
      break;
   case kKaToMu:
      ForceParticleDecay(321,13,1); // K->mu
      break;
   case kWToMuon:
      ForceParticleDecay(  24, 13,1); // W -> mu
      break;
   case kWToCharm:
      ForceParticleDecay(   24, 4,1); // W -> c
      break;
   case kWToCharmToMuon:
      ForceParticleDecay(   24, 4,1); // W -> c
      ForceParticleDecay(  411,13,1); // D+/- -> mu
      ForceParticleDecay(  421,13,1); // D0  -> mu
      ForceParticleDecay(  431,13,1); // D_s  -> mu
      ForceParticleDecay( 4122,13,1); // Lambda_c
      ForceParticleDecay( 4132,13,1); // Xsi_c
      ForceParticleDecay( 4232,13,1); // Sigma_c
      ForceParticleDecay( 4332,13,1); // Omega_c
      break;
   case kZDiMuon:
      ForceParticleDecay(  23, 13,2); // Z -> mu+ mu-
      break;
   case kHadronicD:
      ForceHadronicD();
      break;
   case kPhiKK:
      ForceParticleDecay(333,321,2); // Phi->K+K-
      break;
   case kOmega:
      ForceOmega();
   case kAll:
      break;
   case kNoDecay:
      TPythia6::Instance()->SetMSTJ(21,0);
      break;
   case kNoDecayHeavy: break; // cannot get here: early return above
   case kMaxDecay: break;
   }
}

//______________________________________________________________________________
Float_t TPythia6Decayer::GetPartialBranchingRatio(Int_t ipart)
{
   // Get the partial branching ratio for a particle of type IPART (a
   // PDG code).
   Int_t kc = TPythia6::Instance()->Pycomp(TMath::Abs(ipart));
   // return TPythia6::Instance()->GetBRAT(kc);
   return fBraPart[kc];
}

//______________________________________________________________________________
Float_t TPythia6Decayer::GetLifetime(Int_t kf)
{
   // Get the life-time of a particle of type KF (a PDG code).
   Int_t kc=TPythia6::Instance()->Pycomp(TMath::Abs(kf));
   return TPythia6::Instance()->GetPMAS(kc,4) * 3.3333e-12;
}

//______________________________________________________________________________
void TPythia6Decayer::ReadDecayTable()
{
   // Read in particle data from an ASCII file.   The file name must
   // previously have been set using the member function
   // SetDecayTableFile.
   if (fDecayTableFile.IsNull()) {
      Warning("ReadDecayTable", "No file set");
      return;
   }
   Int_t lun = 15;
   TPythia6::Instance()->OpenFortranFile(lun,
                                         const_cast<char*>(fDecayTableFile.Data()));
   TPythia6::Instance()->Pyupda(3,lun);
   TPythia6::Instance()->CloseFortranFile(lun);
}

// ===================================================================
// BEGIN COMMENT
//
// It would be better if the particle and decay information could be
// read from the current TDatabasePDG instance.
//
// However, it seems to me that some information is missing.  In
// particular
//
//   - The broadning cut-off,
//   - Resonance width
//   - Color charge
//   - MWID (?)
//
// Further more, it's not clear to me at least, what all the
// parameters Pythia needs are.
//
// Code like the below could be used to make a temporary file that
// Pythia could then read in.   Ofcourse, one could also manipulate
// the data structures directly, but that's propably more dangerous.
//
#if 0
void PrintPDG(TParticlePDG* pdg)
{
   TParticlePDG* anti = pdg->AntiParticle();
   const char* antiName = (anti ? anti->GetName() : "");
   Int_t color = 0;
   switch (TMath::Abs(pdg->PdgCode())) {
      case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8: // Quarks
      color = 1; break;
      case 21: // Gluon
      color = 2; break;
      case 1103:
      case 2101: case 2103: case 2203:
      case 3101: case 3103: case 3201: case 3203: case 3303:
      case 4101: case 4103: case 4201: case 4203: case 4301: case 4303: case 4403:
      case 5101: case 5103: case 5201: case 5203: case 5301: case 5303: case 5401:
      case 5403: case 5503:
      // Quark combinations
      color = -1; break;
      case 1000001: case 1000002: case 1000003: case 1000004: case 1000005:
      case 1000006: // super symmetric partners to quars
      color = 1; break;
      case 1000021: // ~g
      color = 2; break;
      case 2000001: case 2000002: case 2000003: case 2000004: case 2000005:
      case 2000006: // R hadrons
      color = 1; break;
      case 3000331: case 3100021: case 3200111: case 3100113: case 3200113:
      case 3300113: case 3400113:
      // Technicolor
      color = 2; break;
      case 4000001: case 4000002:
      color = 1; break;
      case 9900443: case 9900441: case 9910441: case 9900553: case 9900551:
      case 9910551:
      color = 2; break;
   }
   std::cout << std::right
             << " " << std::setw(9) << pdg->PdgCode()
             << "  " << std::left   << std::setw(16) << pdg->GetName()
             << "  " << std::setw(16) << antiName
             << std::right
             << std::setw(3) << Int_t(pdg->Charge())
             << std::setw(3) << color
             << std::setw(3) << (anti ? 1 : 0)
             << std::fixed   << std::setprecision(5)
             << std::setw(12) << pdg->Mass()
             << std::setw(12) << pdg->Width()
             << std::setw(12) << 0 // Broad
             << std::scientific
             << " " << std::setw(13) << pdg->Lifetime()
             << std::setw(3) << 0 // MWID
             << std::setw(3) << pdg->Stable()
             << std::endl;
}

void MakeDecayList()
{
   TDatabasePDG* pdgDB = TDatabasePDG::Instance();
   pdgDB->ReadPDGTable();
   const THashList*    pdgs  = pdgDB->ParticleList();
   TParticlePDG*       pdg   = 0;
   TIter               nextPDG(pdgs);
   while ((pdg = static_cast<TParticlePDG*>(nextPDG()))) {
      // std::cout << "Processing " << pdg->GetName() << std::endl;
      PrintPDG(pdg);

      TObjArray*     decays = pdg->DecayList();
      TDecayChannel* decay  = 0;
      TIter          nextDecay(decays);
      while ((decay = static_cast<TDecayChannel*>(nextDecay()))) {
        // std::cout << "Processing decay number " << decay->Number() << std::endl;
      }
   }
}
#endif
// END COMMENT
// ===================================================================

//______________________________________________________________________________
void TPythia6Decayer::WriteDecayTable()
{
   // write particle data to an ASCII file.   The file name must
   // previously have been set using the member function
   // SetDecayTableFile.
   //
   // Users can use this function to make an initial decay list file,
   // which then can be edited by hand, and re-loaded into the decayer
   // using ReadDecayTable.
   //
   // The file syntax is
   //
   //    particle_list : partcle_data
   //                  | particle_list particle_data
   //                  ;
   //    particle_data : particle_info
   //                  | particle_info '\n' decay_list
   //                  ;
   //    particle_info : See below
   //                  ;
   //    decay_list    : decay_entry
   //                  | decay_list decay_entry
   //                  ;
   //    decay_entry   : See below
   //
   // The particle_info consists of 13 fields:
   //
   //     PDG code             int
   //     Name                 string
   //     Anti-particle name   string  if there's no anti-particle,
   //                                  then this field must be the
   //                                  empty string
   //     Electic charge       int     in units of |e|/3
   //     Color charge         int     in units of quark color charges
   //     Have anti-particle   int     1 of there's an anti-particle
   //                                  to this particle, or 0
   //                                  otherwise
   //     Mass                 float   in units of GeV
   //     Resonance width      float
   //     Max broadning        float
   //     Lifetime             float
   //     MWID                 int     ??? (some sort of flag)
   //     Decay                int     1 if it decays. 0 otherwise
   //
   // The format to write these entries in are
   //
   //    " %9  %-16s  %-16s%3d%3d%3d%12.5f%12.5f%12.5f%13.gf%3d%d\n"
   //
   // The decay_entry consists of 8 fields:
   //
   //     On/Off               int     1 for on, -1 for off
   //     Matrix element type  int
   //     Branching ratio      float
   //     Product 1            int     PDG code of decay product 1
   //     Product 2            int     PDG code of decay product 2
   //     Product 3            int     PDG code of decay product 3
   //     Product 4            int     PDG code of decay product 4
   //     Product 5            int     PDG code of decay product 5
   //
   // The format for these lines are
   //
   //    "          %5d%5d%12.5f%10d%10d%10d%10d%10d\n"
   //
   if (fDecayTableFile.IsNull()) {
      Warning("ReadDecayTable", "No file set");
      return;
   }
   Int_t lun = 15;
   TPythia6::Instance()->OpenFortranFile(lun,
                                         const_cast<char*>(fDecayTableFile.Data()));
   TPythia6::Instance()->Pyupda(1,lun);
   TPythia6::Instance()->CloseFortranFile(lun);
}

//______________________________________________________________________________
Int_t TPythia6Decayer::CountProducts(Int_t channel, Int_t particle)
{
   // Count number of decay products
   Int_t np = 0;
   for (Int_t i = 1; i <= 5; i++)
      if (TMath::Abs(TPythia6::Instance()->GetKFDP(channel,i)) == particle) np++;
   return np;
}

//______________________________________________________________________________
void TPythia6Decayer::ForceHadronicD()
{
   // Force golden D decay modes
   const Int_t kNHadrons = 4;
   Int_t channel;
   Int_t hadron[kNHadrons] = {411,  421, 431, 4112};

   // for D+ -> K0* (-> K- pi+) pi+
   Int_t iKstar0     =  313;
   Int_t iKstarbar0  = -313;
   Int_t products[2] = {kKPlus, kPiMinus}, mult[2] = {1, 1};
   ForceParticleDecay(iKstar0, products, mult, 2);

   // for Ds -> Phi pi+
   Int_t iPhi = 333;
   ForceParticleDecay(iPhi,kKPlus,2); // Phi->K+K-

   Int_t decayP1[kNHadrons][3] = {
      {kKMinus, kPiPlus,    kPiPlus},
      {kKMinus, kPiPlus,    0      },
      {kKPlus , iKstarbar0, 0     },
      {-1     , -1        , -1        }
   };
   Int_t decayP2[kNHadrons][3] = {
      {iKstarbar0, kPiPlus, 0   },
      {-1        , -1     , -1  },
      {iPhi      , kPiPlus, 0  },
      {-1        , -1     , -1  }
   };

   TPythia6* pyth = TPythia6::Instance();
   for (Int_t ihadron = 0; ihadron < kNHadrons; ihadron++) {
      Int_t kc = pyth->Pycomp(hadron[ihadron]);
      pyth->SetMDCY(kc,1,1);
      Int_t ifirst = pyth->GetMDCY(kc,2);
      Int_t ilast  = ifirst + pyth->GetMDCY(kc,3)-1;

      for (channel = ifirst; channel <= ilast; channel++) {
         if ((pyth->GetKFDP(channel,1) == decayP1[ihadron][0] &&
            pyth->GetKFDP(channel,2) == decayP1[ihadron][1] &&
            pyth->GetKFDP(channel,3) == decayP1[ihadron][2] &&
            pyth->GetKFDP(channel,4) == 0) ||
           (pyth->GetKFDP(channel,1) == decayP2[ihadron][0] &&
            pyth->GetKFDP(channel,2) == decayP2[ihadron][1] &&
            pyth->GetKFDP(channel,3) == decayP2[ihadron][2] &&
            pyth->GetKFDP(channel,4) == 0)) {
            pyth->SetMDME(channel,1,1);
         } else {
            pyth->SetMDME(channel,1,0);
            fBraPart[kc] -= pyth->GetBRAT(channel);
         } // selected channel ?
      } // decay channels
   } // hadrons
}

//______________________________________________________________________________
void TPythia6Decayer::ForceParticleDecay(Int_t particle, Int_t product, Int_t mult)
{
   //
   //  Force decay of particle into products with multiplicity mult
   TPythia6* pyth = TPythia6::Instance();

   Int_t kc =  pyth->Pycomp(particle);
   pyth->SetMDCY(kc,1,1);

   Int_t ifirst = pyth->GetMDCY(kc,2);
   Int_t ilast  = ifirst + pyth->GetMDCY(kc,3)-1;
   fBraPart[kc] = 1;

   //
   //  Loop over decay channels
   for (Int_t channel= ifirst; channel <= ilast; channel++) {
      if (CountProducts(channel,product) >= mult) {
         pyth->SetMDME(channel,1,1);
      } else {
         pyth->SetMDME(channel,1,0);
         fBraPart[kc]-=pyth->GetBRAT(channel);
      }
   }
}

//______________________________________________________________________________
void TPythia6Decayer::ForceParticleDecay(Int_t particle, Int_t* products,
                                         Int_t* mult, Int_t npart)
{
   //
   //  Force decay of particle into products with multiplicity mult
   TPythia6* pyth = TPythia6::Instance();

   Int_t kc     = pyth->Pycomp(particle);
   pyth->SetMDCY(kc,1,1);
   Int_t ifirst = pyth->GetMDCY(kc,2);
   Int_t ilast  = ifirst+pyth->GetMDCY(kc,3)-1;
   fBraPart[kc] = 1;
   //
   //  Loop over decay channels
   for (Int_t channel = ifirst; channel <= ilast; channel++) {
      Int_t nprod = 0;
      for (Int_t i = 0; i < npart; i++)
         nprod += (CountProducts(channel, products[i]) >= mult[i]);
      if (nprod)
         pyth->SetMDME(channel,1,1);
      else {
         pyth->SetMDME(channel,1,0);
         fBraPart[kc] -= pyth->GetBRAT(channel);
      }
   }
}

//______________________________________________________________________________
void TPythia6Decayer::ForceOmega()
{
   // Force Omega -> Lambda K- Decay
   TPythia6* pyth = TPythia6::Instance();

   Int_t kc     = pyth->Pycomp(3334);
   pyth->SetMDCY(kc,1,1);
   Int_t ifirst = pyth->GetMDCY(kc,2);
   Int_t ilast  = ifirst + pyth->GetMDCY(kc,3)-1;
   for (Int_t channel = ifirst; channel <= ilast; channel++) {
      if (pyth->GetKFDP(channel,1) == kLambda0 &&
         pyth->GetKFDP(channel,2) == kKMinus  &&
         pyth->GetKFDP(channel,3) == 0)
         pyth->SetMDME(channel,1,1);
      else
         pyth->SetMDME(channel,1,0);
      // selected channel ?
   } // decay channels
}
