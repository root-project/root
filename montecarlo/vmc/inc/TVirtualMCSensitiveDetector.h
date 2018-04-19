// @(#)root/vmc:$Id$
// Authors: Ivana Hrivnacova 19/04/2018

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2018 ALICE Experiment at CERN.                          *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualMCSensitiveDetector
#define ROOT_TVirtualMCSensitiveDetector

// Class TVirtualMCSensitiveDetector
// ---------------------------------
// Interface to a user defined particles stack.
//

#include "TNamed.h"

class TParticle;

class TVirtualMCSensitiveDetector : public TNamed {

public:
   // Constructor
   TVirtualMCSensitiveDetector(const char *name, const char *title = "");
   TVirtualMCSensitiveDetector(const TString &name, const TString &title = "");

   // Destructor
   virtual ~TVirtualMCSensitiveDetector();

   /// Initialize detector.
   /// Called at initialization of geometry before MCApplication::InitGeometry().
   virtual void Initialize() = 0;

   /// Process hits.
   /// Called at each step when track pass through the associated volume
   virtual void ProcessHits() = 0;

   /// End of event.
   /// Called at end of event before MCApplication::FinishEvent().
   virtual void EndOfEvent() = 0;

protected:
   // Default constructor
   TVirtualMCSensitiveDetector();
   // Copy constructor
   TVirtualMCSensitiveDetector(const TVirtualMCSensitiveDetector &rhs);
   // Assignment constructor
   TVirtualMCSensitiveDetector &operator=(const TVirtualMCSensitiveDetector &rhs);

   ClassDef(TVirtualMCSensitiveDetector, 1) // Interface to a user sensitive detector
};

#endif // ROOT_TVirtualMCSensitiveDetector
