// @(#)root/pythia8:$Name$:$Id$
// Author: Andreas Morsch   04/07/2008

/**************************************************************************
 * Copyright(c) 1998-2008, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

#include <TLorentzVector.h>
#include <TPythia8.h>
#include "TPythia8Decayer.h"

ClassImp(TPythia8Decayer)

//___________________________________________________________________________
TPythia8Decayer::TPythia8Decayer():
  fPythia8(new TPythia8()),
  fDebug(0)
{
   //constructor
   fPythia8->Pythia8()->readString("SoftQCD:elastic = on");
   fPythia8->Pythia8()->init();
}

//___________________________________________________________________________
void TPythia8Decayer::Init()
{
   // Initialize the decayer
}

//___________________________________________________________________________
void TPythia8Decayer::Decay(Int_t pdg, TLorentzVector* p)
{
   // Decay a single particle
   ClearEvent();
   AppendParticle(pdg, p);
   Int_t idPart = fPythia8->Pythia8()->event[0].id();
   fPythia8->Pythia8()->particleData.mayDecay(idPart,kTRUE);
   fPythia8->Pythia8()->moreDecays();
   if (fDebug > 0) fPythia8->EventListing();
}

//___________________________________________________________________________
Int_t TPythia8Decayer::ImportParticles(TClonesArray *particles)
{
   //import the decay products into particles array
   return (fPythia8->ImportParticles(particles, "All"));
}

//___________________________________________________________________________
void TPythia8Decayer::SetForceDecay(Int_t /*type*/)
{
   // Set forced decay mode
   printf("SetForceDecay not yet implemented !\n");
}
//___________________________________________________________________________
void TPythia8Decayer::ForceDecay()
{
   // ForceDecay not yet implemented
   printf("ForceDecay not yet implemented !\n");
}
//___________________________________________________________________________
Float_t TPythia8Decayer::GetPartialBranchingRatio(Int_t /*ipart*/)
{
   return 0.0;
}
//___________________________________________________________________________
Float_t TPythia8Decayer::GetLifetime(Int_t pdg) 
{
   //return lifetime in seconds of teh particle with PDG number pdg
   return (fPythia8->Pythia8()->particleData.tau0(pdg) * 3.3333e-12) ;
}

//___________________________________________________________________________
void    TPythia8Decayer::ReadDecayTable()
{
   //to read a decay table (not yet implemented)
}


//___________________________________________________________________________
void TPythia8Decayer::AppendParticle(Int_t pdg, TLorentzVector* p)
{
   // Append a particle to the stack
   fPythia8->Pythia8()->event.append(pdg, 11, 0, 0, p->Px(), p->Py(), p->Pz(), p->E(), p->M());   
}


//___________________________________________________________________________
void TPythia8Decayer::ClearEvent()
{
   // Clear the event stack
   fPythia8->Pythia8()->event.clear();
}

