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

#ifndef ROOT_TMCManager
#define ROOT_TMCManager
//
// Class TMCManager
// ---------------------------
// manager class for handling multiple TVirtualMC engines.
//

#include <functional>
#include <memory>

#include "TMCtls.h"
#include "TGeoMCBranchArrayContainer.h"
#include "TMCParticleStatus.h"
#include "TGeoManager.h"
#include "TVirtualMC.h"

class TVirtualMC;
class TVirtualMCApplication;
class TParticle;
class TVirtualMCStack;
class TMCManagerStack;

class TMCManager {

   friend class TVirtualMCApplication;

public:
   /// Default constructor
   TMCManager();

   /// Destructor
   virtual ~TMCManager();

   /// Static access method
   static TMCManager *Instance();

   //
   // Methods to manage multiple engines
   //

   /// A TVirtualMC will register itself via this method during construction
   /// if a TMCManager was instanciated before.
   /// The TMCManager will assign an ID to the engines.
   void Register(TVirtualMC *engine);

   /// The user application will register itself via this method when the
   /// manager was requested.
   void Register(TVirtualMCApplication *application);

   /// Return the number of registered engines.
   Int_t NEngines() const;

   /// Get registered engine pointers
   void GetEngines(std::vector<TVirtualMC *> &engines) const;

   /// Get an engine pointer by ID
   TVirtualMC *GetEngine(Int_t id) const;

   /// Get engine ID by its name
   Int_t GetEngineId(const char *name) const;

   /// Get the current engine pointer
   TVirtualMC *GetCurrentEngine() const;

   /// Connect a pointer which is updated whenever the engine is changed
   void ConnectEnginePointer(TVirtualMC **mc);

   /// Connect a pointer which is updated whenever the engine is changed
   void ConnectEnginePointer(TVirtualMC *&mc);

   //
   // Stack related methods
   //

   /// Set user stack
   void SetUserStack(TVirtualMCStack *stack);

   /// User interface to forward particle to specifiic engine.
   /// It is assumed that the TParticle is owned by the user. It will not be
   /// modified by the TMCManager.
   void ForwardTrack(Int_t toBeDone, Int_t trackId, Int_t parentId, TParticle *particle, Int_t engineId);

   /// User interface to forward particle to specifiic engine.
   /// It is assumed that the TParticle is owned by the user. It will not be
   /// modified by the TMCManager.
   /// Assume current engine Id
   void ForwardTrack(Int_t toBeDone, Int_t trackId, Int_t parentId, TParticle *particle);

   /// Transfer track from current engine to engine with engineTargetId
   void TransferTrack(Int_t engineTargetId);

   /// Transfer track from current engine to target engine mc
   void TransferTrack(TVirtualMC *mc);

   /// Try to restore geometry for a given track
   Bool_t RestoreGeometryState(Int_t trackId, Bool_t checkTrackIdRange = kTRUE);

   /// Try to restore geometry for the track currently set
   Bool_t RestoreGeometryState();

   //
   // Steering and control
   //

   /// Apply something to all engines
   template <typename F>
   void Apply(F engineLambda)
   {
      for (auto &mc : fEngines) {
         // We never know whether static method TVirtualMC::GetMC() is used in any way so update before calling the
         // lambda.
         UpdateEnginePointers(mc);
         engineLambda(mc);
      }
   }

   /// Initialize engines
   void Init();
   /// Further specific initialization
   template <typename F>
   void Init(F initFunction)
   {
      if (fIsInitializedUser) {
         return;
      }
      Init();
      for (auto &mc : fEngines) {
         // Set to current engine and call user init procedure
         UpdateEnginePointers(mc);
         initFunction(mc);
      }
      fIsInitializedUser = kTRUE;
   }

   /// Run the event loop
   void Run(Int_t nEvents);

private:
   /// Do necessary steps before an event is triggered
   void PrepareNewEvent();
   /// Find the  next engine
   Bool_t GetNextEngine();
   /// Update all engine pointers connected to the TMCManager
   void UpdateEnginePointers(TVirtualMC *mc);
   /// Terminate a run in all engines
   void TerminateRun();

private:
   // static data members
#if !defined(__CINT__)
   static TMCThreadLocal TMCManager *fgInstance; ///< Singleton instance
#else
   static TMCManager *fgInstance; ///< Singleton instance
#endif

   /// Pointer to user application
   TVirtualMCApplication *fApplication;
   /// Pointer to current engine
   TVirtualMC *fCurrentEngine;
   /// Collecting pointers to all instanciated TVirtualMCs
   std::vector<TVirtualMC *> fEngines;
   /// Stacks connected to engines
   std::vector<std::unique_ptr<TMCManagerStack>> fStacks;
   /// All tracks (persistent)
   std::vector<TParticle *> fParticles;
   /// All particles' status (persistent)
   std::vector<std::unique_ptr<TMCParticleStatus>> fParticlesStatus;
   /// Total number of primaries ever pushed
   Int_t fTotalNPrimaries;
   /// Total number of tracks ever pushed
   Int_t fTotalNTracks;
   /// Connected engine pointers which will be updated everytime the current
   /// engine changes
   std::vector<TVirtualMC **> fConnectedEnginePointers;
   /// Pointer to user stack
   TVirtualMCStack *fUserStack;
   /// Pointer to cache with geometry states
   TGeoMCBranchArrayContainer fBranchArrayContainer;
   /// Flag if engines are initilaized
   Bool_t fIsInitialized;
   /// Flag if specific initialization for engines was done
   Bool_t fIsInitializedUser;

   ClassDef(TMCManager, 0)
};

#endif
