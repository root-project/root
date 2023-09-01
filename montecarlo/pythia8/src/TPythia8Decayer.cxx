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

/** \class TPythia8Decayer
    \ingroup pythia8

This class implements the TVirtualMCDecayer interface using TPythia8.

Author: Andreas Morsch   04/07/2008
*/

#include "TLorentzVector.h"
#include "TPythia8.h"
#include "TPythia8Decayer.h"

ClassImp(TPythia8Decayer);

////////////////////////////////////////////////////////////////////////////////
///constructor

TPythia8Decayer::TPythia8Decayer():
  fPythia8(new TPythia8()),
  fDebug(0)
{
   fPythia8->Pythia8()->readString("SoftQCD:elastic = on");
   fPythia8->Pythia8()->init();
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the decayer

void TPythia8Decayer::Init()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Decay a single particle

void TPythia8Decayer::Decay(Int_t pdg, TLorentzVector* p)
{
   ClearEvent();
   AppendParticle(pdg, p);
   Int_t idPart = fPythia8->Pythia8()->event[0].id();
   fPythia8->Pythia8()->particleData.mayDecay(idPart,kTRUE);
   fPythia8->Pythia8()->moreDecays();
   if (fDebug > 0) fPythia8->EventListing();
}

////////////////////////////////////////////////////////////////////////////////
///import the decay products into particles array

Int_t TPythia8Decayer::ImportParticles(TClonesArray *particles)
{
   return (fPythia8->ImportParticles(particles, "All"));
}

////////////////////////////////////////////////////////////////////////////////
/// Set forced decay mode

void TPythia8Decayer::SetForceDecay(Int_t /*type*/)
{
   printf("SetForceDecay not yet implemented !\n");
}
////////////////////////////////////////////////////////////////////////////////
/// ForceDecay not yet implemented

void TPythia8Decayer::ForceDecay()
{
   printf("ForceDecay not yet implemented !\n");
}
////////////////////////////////////////////////////////////////////////////////

Float_t TPythia8Decayer::GetPartialBranchingRatio(Int_t /*ipart*/)
{
   return 0.0;
}
////////////////////////////////////////////////////////////////////////////////
///return lifetime in seconds of teh particle with PDG number pdg

Float_t TPythia8Decayer::GetLifetime(Int_t pdg)
{
   return (fPythia8->Pythia8()->particleData.tau0(pdg) * 3.3333e-12) ;
}

////////////////////////////////////////////////////////////////////////////////
///to read a decay table (not yet implemented)

void    TPythia8Decayer::ReadDecayTable()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Append a particle to the stack

void TPythia8Decayer::AppendParticle(Int_t pdg, TLorentzVector* p)
{
   fPythia8->Pythia8()->event.append(pdg, 11, 0, 0, p->Px(), p->Py(), p->Pz(), p->E(), p->M());
}


////////////////////////////////////////////////////////////////////////////////
/// Clear the event stack

void TPythia8Decayer::ClearEvent()
{
   fPythia8->Pythia8()->event.clear();
}

