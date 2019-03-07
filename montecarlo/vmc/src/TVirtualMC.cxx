// @(#)root/vmc:$Id$
// Authors: Ivana Hrivnacova, Rene Brun , Federico Carminati 13/04/2002

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2002, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualMC.h"
#include "TError.h"

/** \class TVirtualMC
    \ingroup vmc

Abstract Monte Carlo interface

Virtual MC provides a virtual interface to Monte Carlo.
It enables the user to build a virtual Monte Carlo application
independent of any actual underlying Monte Carlo implementation itself.

A user will have to implement a class derived from the abstract
Monte Carlo application class, and provide functions like
ConstructGeometry(), BeginEvent(), FinishEvent(), ... .
The concrete Monte Carlo (Geant3, Geant4) is selected at run time -
when processing a ROOT macro where the concrete Monte Carlo is instantiated.
*/

TMCThreadLocal TVirtualMC *TVirtualMC::fgMC = nullptr;
////////////////////////////////////////////////////////////////////////////////
///
/// Standard constructor
///

TVirtualMC::TVirtualMC(const char *name, const char *title, Bool_t /*isRootGeometrySupported*/)
   : TNamed(name, title), fApplication(nullptr), fId(0), fStack(nullptr), fManagerStack(nullptr), fDecayer(nullptr),
     fRandom(nullptr), fMagField(nullptr), fUseExternalGeometryConstruction(kFALSE),
     fUseExternalParticleGeneration(kFALSE)
{
   fApplication = TVirtualMCApplication::Instance();

   if (fApplication) {
      fApplication->Register(this);
   } else {
      ::Fatal("TVirtualMC::TVirtualMC", "No user MC application is defined.");
   }
   fgMC = this;
   fRandom = gRandom;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Default constructor
///

TVirtualMC::TVirtualMC()
   : TNamed(), fApplication(nullptr), fId(0), fStack(nullptr), fManagerStack(nullptr), fDecayer(nullptr),
     fRandom(nullptr), fMagField(nullptr), fUseExternalGeometryConstruction(kFALSE),
     fUseExternalParticleGeneration(kFALSE)
{
}

////////////////////////////////////////////////////////////////////////////////
///
/// Destructor
///

TVirtualMC::~TVirtualMC()
{
   fgMC = nullptr;
}

//
// methods
//

////////////////////////////////////////////////////////////////////////////////
///
/// Static access method
///

TVirtualMC *TVirtualMC::GetMC()
{
   return fgMC;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Set particles stack.
///

void TVirtualMC::SetStack(TVirtualMCStack *stack)
{
   fStack = stack;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Set external decayer.
///

void TVirtualMC::SetExternalDecayer(TVirtualMCDecayer *decayer)
{
   fDecayer = decayer;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Set random number generator.
///

void TVirtualMC::SetRandom(TRandom *random)
{
   gRandom = random;
   fRandom = random;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Set magnetic field.
///

void TVirtualMC::SetMagField(TVirtualMagField *field)
{
   fMagField = field;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Process one event (backwards compatibility)
///

void TVirtualMC::ProcessEvent()
{
   Warning("ProcessEvent", "Not implemented.");
}

////////////////////////////////////////////////////////////////////////////////
///
/// Process one event (backwards compatibility)
///

void TVirtualMC::ProcessEvent(Int_t eventId)
{
   ProcessEvent(eventId, kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
///
/// Return the current step number
///

Int_t TVirtualMC::StepNumber() const
{
   Warning("StepNumber", "Not implemented.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Get the current weight
///

Double_t TVirtualMC::TrackWeight() const
{
   Warning("Weight", "Not implemented.");
   return 1.;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Get the current polarization
///

void TVirtualMC::TrackPolarization(Double_t &polX, Double_t &polY, Double_t &polZ) const
{
   Warning("Polarization", "Not implemented.");
   polX = 0.;
   polY = 0.;
   polZ = 0.;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Get the current polarization
///

void TVirtualMC::TrackPolarization(TVector3 &pol) const
{
   Warning("Polarization", "Not implemented.");
   pol[0] = 0.;
   pol[1] = 0.;
   pol[2] = 0.;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Set the VMC id
///

void TVirtualMC::SetId(UInt_t id)
{
   fId = id;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Set container holding additional information for transported TParticles
///
void TVirtualMC::SetManagerStack(TMCManagerStack *stack)
{
   fManagerStack = stack;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Disables internal dispatch to TVirtualMCApplication::ConstructGeometry()
/// and hence rely on geometry construction being trigeered from outside.
///

void TVirtualMC::SetExternalGeometryConstruction(Bool_t value)
{
   fUseExternalGeometryConstruction = value;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Disables internal dispatch to TVirtualMCApplication::ConstructGeometry()
/// and hence rely on geometry construction being trigeered from outside.
///

void TVirtualMC::SetExternalParticleGeneration(Bool_t value)
{
   fUseExternalParticleGeneration = value;
}

////////////////////////////////////////////////////////////////////////////////
///
/// An interruptible event can be paused and resumed at any time. It must not
/// call TVirtualMCApplication::BeginEvent() and ::FinishEvent()
/// Further, when tracks are popped from the TVirtualMCStack it must be
/// checked whether these are new tracks or whether they have been
/// transported up to their current point.
///

void TVirtualMC::ProcessEvent(Int_t eventId, Bool_t isInterruptible)
{
   const char *interruptibleText = isInterruptible ? "interruptible" : "non-interruptible";
   Warning("ProcessInterruptibleEvent", "Process %s event %i. Not implemented.", interruptibleText, eventId);
}

////////////////////////////////////////////////////////////////////////////////
///
/// That triggers stopping the transport of the current track without dispatching
/// to common routines like TVirtualMCApplication::PostTrack() etc.
///

void TVirtualMC::InterruptTrack()
{
   Warning("InterruptTrack", "Not implemented.");
}
