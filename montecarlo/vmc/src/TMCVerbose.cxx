// @(#)root/vmc:$Id$
// Author: Ivana Hrivnacova, 27/03/2002

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2002, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TVirtualMC.h"
#include "TVirtualMCStack.h"
#include "TDatabasePDG.h"
#include "TParticlePDG.h"
#include "TArrayI.h"

#include "TMCVerbose.h"

//______________________________________________________________________________
//
// Class for printing detailed info from MC application.
// Defined levels:
//  0  no output
//  1  info up to event level
//  2  info up to tracking level
//  3  detailed info for each step
//______________________________________________________________________________


ClassImp(TMCVerbose)

//_____________________________________________________________________________
TMCVerbose::TMCVerbose(Int_t level)
  : TObject(),
    fLevel(level),
    fStepNumber(0)
{
// Standard constructor
// ---

}

//_____________________________________________________________________________
TMCVerbose::TMCVerbose()
  : TObject(),
    fLevel(0),
    fStepNumber(0)
{
// Default constructor
// ---
}

//_____________________________________________________________________________
TMCVerbose::~TMCVerbose()
{
// Destructor
// ---
}

//
// private methods
//

//_____________________________________________________________________________
void TMCVerbose::PrintBanner() const
{
// Prints banner for track information
// ---

   std::cout << std::endl;
   for (Int_t i=0; i<10; i++) std::cout << "**********";
   std::cout << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::PrintTrackInfo() const
{
// Prints track information
// ---

   // Particle
   //
   std::cout << "  Particle = ";
   TParticlePDG* particle = TDatabasePDG::Instance()->GetParticle(gMC->TrackPid());
   if (particle)
      std::cout << particle->GetName() << "  ";
   else
      std::cout << "unknown" << "  ";

   // Track ID
   //
   std::cout << "   Track ID = " << gMC->GetStack()->GetCurrentTrackNumber() << "  ";

   // Parent ID
   //
   std::cout << "   Parent ID = " << gMC->GetStack()->GetCurrentParentTrackNumber();
}

//_____________________________________________________________________________
void TMCVerbose::PrintStepHeader() const
{
// Prints the header for stepping information
// ---

   std::cout << "Step#     "
        << "X(cm)    "
        << "Y(cm)    "
        << "Z(cm)  "
        << "KinE(MeV)   "
        << "dE(MeV) "
        << "Step(cm) "
        << "TrackL(cm) "
        << "Volume  "
        << "Process "
        << std::endl;
}

//
// public methods
//

//_____________________________________________________________________________
void TMCVerbose::InitMC()
{
// Initialize MC info.
// ---

   if (fLevel>0)
      std::cout << "--- Init MC " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::RunMC(Int_t nofEvents)
{
// MC run info.
// ---

   if (fLevel>0)
      std::cout << "--- Run MC for " << nofEvents << " events" << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::FinishRun()
{
// Finish MC run info.
// ---

   if (fLevel>0)
      std::cout << "--- Finish Run MC " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::ConstructGeometry()
{
// Construct geometry info
// ---

   if (fLevel>0)
      std::cout << "--- Construct geometry " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::ConstructOpGeometry()
{
// Construct geometry for optical physics info
// ---

   if (fLevel>0)
      std::cout << "--- Construct geometry for optical processes" << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::InitGeometry()
{
// Initialize geometry info
// ---

   if (fLevel>0)
      std::cout << "--- Init geometry " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::AddParticles()
{
// Add particles info
// ---

   if (fLevel>0)
      std::cout << "--- Add particles " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::AddIons()
{
// Add ions info
// ---

   if (fLevel>0)
      std::cout << "--- Add ions " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::GeneratePrimaries()
{
// Generate primaries info
// ---

   if (fLevel>0)
      std::cout << "--- Generate primaries " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::BeginEvent()
{
// Begin event info
// ---

   if (fLevel>0)
      std::cout << "--- Begin event " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::BeginPrimary()
{
// Begin of a primary track info
// ---

   if (fLevel>1)
      std::cout << "--- Begin primary " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::PreTrack()
{
// Begin of each track info
// ---

   if (fLevel>2) {
      PrintBanner();
      PrintTrackInfo();
      PrintBanner();
      PrintStepHeader();

      fStepNumber = 0;

      return;
   }

   if (fLevel>1)
      std::cout << "--- Pre track " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::Stepping()
{
// Stepping info
// ---

   if (fLevel>2) {

#if __GNUC__ >= 3
      std::cout << std::fixed;
#endif

      // Step number
      //
      std::cout << "#" << std::setw(4) << fStepNumber++ << "  ";

      // Position
      //
      Double_t x, y, z;
      gMC->TrackPosition(x, y, z);
      std::cout << std::setw(8) << std::setprecision(3) << x << " "
           << std::setw(8) << std::setprecision(3) << y << " "
           << std::setw(8) << std::setprecision(3) << z << "  ";

      // Kinetic energy
      //
      Double_t px, py, pz, etot;
      gMC->TrackMomentum(px, py, pz, etot);
      Double_t ekin = etot - gMC->TrackMass();
      std::cout << std::setw(9) << std::setprecision(4) << ekin*1e03 << " ";

      // Energy deposit
      //
      std::cout << std::setw(9) << std::setprecision(4) << gMC->Edep()*1e03 << " ";

      // Step length
      //
      std::cout << std::setw(8) << std::setprecision(3) << gMC->TrackStep() << " ";

      // Track length
      //
      std::cout << std::setw(8) << std::setprecision(3) << gMC->TrackLength() << "     ";

      // Volume
      //
      if (gMC->CurrentVolName() != 0)
         std::cout << std::setw(4) << gMC->CurrentVolName() << "  ";
      else
         std::cout << std::setw(4) << "None"  << "  ";

      // Process
      //
      TArrayI processes;
      Int_t nofProcesses = gMC->StepProcesses(processes);
      if (nofProcesses > 0)
         std::cout << TMCProcessName[processes[nofProcesses-1]];

      std::cout << std::endl;
   }
}

//_____________________________________________________________________________
void TMCVerbose::PostTrack()
{
// Finish of each track info
// ---

   if (fLevel==2)
      std::cout << "--- Post track " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::FinishPrimary()
{
// Finish of a primary track info
// ---

   if (fLevel==2)
      std::cout << "--- Finish primary " << std::endl;
}

//_____________________________________________________________________________
void TMCVerbose::FinishEvent()
{
// Finish of an event info
// ---

   if (fLevel>0)
      std::cout << "--- Finish event " << std::endl;
}

