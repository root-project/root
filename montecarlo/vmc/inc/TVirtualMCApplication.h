// @(#)root/vmc:$Id$
// Author: Ivana Hrivnacova, 23/03/2002

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2002, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualMCApplication
#define ROOT_TVirtualMCApplication
//
// Class TVirtualMCApplication
// ---------------------------
// Interface to a user Monte Carlo application.
//

#include "TNamed.h"
#include "TMath.h"

#include "TMCtls.h"

class TVirtualMC;
class TMCManager;

class TVirtualMCApplication : public TNamed {

public:
   /// Standard constructor
   TVirtualMCApplication(const char *name, const char *title);

   /// Default constructor
   TVirtualMCApplication();

   /// Destructor
   virtual ~TVirtualMCApplication();

   /// Static access method
   static TVirtualMCApplication *Instance();

   //
   // methods
   //

   /// Request a TMCManager which is required if multiple engines should be run
   void RequestMCManager();

   /// Register the an engine.
   void Register(TVirtualMC *mc);

   /// Return the transport engine registered to this application
   TVirtualMC *GetMC() const;

   /// Construct user geometry
   virtual void ConstructGeometry() = 0;

   /// Misalign user geometry (optional)
   virtual Bool_t MisalignGeometry() { return kFALSE; }

   /// Define parameters for optical processes (optional)
   virtual void ConstructOpGeometry() {}

   /// Define sensitive detectors (optional)
   virtual void ConstructSensitiveDetectors() {}

   /// Initialize geometry
   /// (Usually used to define sensitive volumes IDs)
   virtual void InitGeometry() = 0;

   /// Add user defined particles (optional)
   virtual void AddParticles() {}

   /// Add user defined ions (optional)
   virtual void AddIons() {}

   /// Generate primary particles
   virtual void GeneratePrimaries() = 0;

   /// Define actions at the beginning of the event
   virtual void BeginEvent() = 0;

   /// Define actions at the beginning of the primary track
   virtual void BeginPrimary() = 0;

   /// Define actions at the beginning of each track
   virtual void PreTrack() = 0;

   /// Define action at each step
   virtual void Stepping() = 0;

   /// Define actions at the end of each track
   virtual void PostTrack() = 0;

   /// Define actions at the end of the primary track
   virtual void FinishPrimary() = 0;

   /// Define actions at the end of the event
   virtual void FinishEvent() = 0;

   /// Define maximum radius for tracking (optional)
   virtual Double_t TrackingRmax() const { return DBL_MAX; }

   /// Define maximum z for tracking (optional)
   virtual Double_t TrackingZmax() const { return DBL_MAX; }

   /// Calculate user field \a b at point \a x
   virtual void Field(const Double_t *x, Double_t *b) const;

   /// Define action at each step for Geane
   virtual void GeaneStepping() { ; }

   // Functions for multi-threading applications
   /// Clone MC application on worker
   virtual TVirtualMCApplication *CloneForWorker() const { return 0; }

   /// Const Initialize MC application on worker  - now deprecated
   /// Use new non-const InitOnWorker()  instead
   virtual void InitForWorker() const {}
   /// Const Define actions at the beginning of the worker run if needed - now deprecated
   /// Use new non-const BeginRunOnWorker() instead
   virtual void BeginWorkerRun() const {}
   /// Const Define actions at the end of the worker run if needed - now deprecated
   /// Use new non-const FinishRunOnWorker() instead
   virtual void FinishWorkerRun() const {}

   /// Initialize MC application on worker
   virtual void InitOnWorker() {}
   /// Define actions at the beginning of the worker run if needed
   virtual void BeginRunOnWorker() {}
   /// Define actions at the end of the worker run if needed
   virtual void FinishRunOnWorker() {}
   /// Merge the data accumulated on workers to the master if needed
   virtual void Merge(TVirtualMCApplication * /*localMCApplication*/) {}

protected:
   /// The current transport engine in use. In case of a multi-run the TMCManager
   /// will update this whenever the engine changes.
   TVirtualMC *fMC;

   /// Pointer to requested TMCManager which will only be instantiated by a call
   /// to RequestMCManager()
   TMCManager *fMCManager;

private:
   // static data members
#if !defined(__CINT__)
   static TMCThreadLocal TVirtualMCApplication
      *fgInstance; ///< Singleton instance
#else
   static TVirtualMCApplication *fgInstance; ///< Singleton instance
#endif
                   /// Forbid multithreading mode if multi run via global static flag
   static Bool_t fLockMultiThreading;

   ClassDef(TVirtualMCApplication, 1) // Interface to MonteCarlo application
};

inline void TVirtualMCApplication::Field(const Double_t * /*x*/, Double_t *b) const
{
   // No magnetic field
   b[0] = 0;
   b[1] = 0;
   b[2] = 0;
}

#endif // ROOT_TVirtualMCApplication
