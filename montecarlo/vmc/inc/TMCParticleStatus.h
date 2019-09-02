// @(#)root/vmc:$Id$
// Authors: Benedikt Volkel 07/03/2019

/*************************************************************************
 * Copyright (C) 2019, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2019, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMCParticleStatus
#define ROOT_TMCParticleStatus

// Class TMCParticleStatus
// ---------------------
// additional information on the current status of a TParticle
//

#include <iostream>

#include "TVector3.h"
#include "TLorentzVector.h"
#include "TParticle.h"
#include "TError.h"

struct TMCParticleStatus {

   /// Default constructor
   TMCParticleStatus() = default;

   /// Use TParticle information as a starting point
   void InitFromParticle(const TParticle *particle)
   {
      particle->ProductionVertex(fPosition);
      particle->Momentum(fMomentum);
      particle->GetPolarisation(fPolarization);
      fWeight = particle->GetWeight();
   }

   virtual ~TMCParticleStatus() = default;

   //
   // verbosity
   //

   /// Print all info at once
   void Print() const
   {
      ::Info("Print", "Status of track");
      std::cout << "\t"
                << "ID: " << fId << "\n"
                << "\t"
                << "parentID: " << fParentId << "\n"
                << "\t"
                << "weight: " << fWeight << "\n"
                << "\t"
                << "geo state index: " << fGeoStateIndex << "\n"
                << "\t"
                << "step number: " << fStepNumber << "\n"
                << "\t"
                << "track length: " << fTrackLength << "\n"
                << "\t"
                << "position" << std::endl;
      fPosition.Print();
      std::cout << "\t"
                << "momentum" << std::endl;
      fMomentum.Print();
      std::cout << "\t"
                << "polarization" << std::endl;
      fPolarization.Print();
   }

   /// Number of steps
   Int_t fStepNumber = 0;
   /// Track length
   Double_t fTrackLength = 0.;
   ///  position
   TLorentzVector fPosition;
   ///  momentum
   TLorentzVector fMomentum;
   ///  polarization
   TVector3 fPolarization;
   ///  weight
   Double_t fWeight = 1.;
   ///  geo state cache
   UInt_t fGeoStateIndex = 0;
   /// Unique ID assigned by the user
   Int_t fId = -1;
   /// Unique ID assigned by the user
   Int_t fParentId = -1;

private:
   /// Copying kept private
   TMCParticleStatus(const TMCParticleStatus &);
   /// Assignement kept private
   TMCParticleStatus &operator=(const TMCParticleStatus &);

   ClassDef(TMCParticleStatus, 1)
};

#endif /* ROOT_TMCParticleStatus */
