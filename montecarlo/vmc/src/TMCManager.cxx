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

#include "TError.h"
#include "TVector3.h"
#include "TLorentzVector.h"
#include "TParticle.h"
#include "TGeoBranchArray.h"
#include "TGeoNavigator.h"

#include "TVirtualMCApplication.h"
#include "TVirtualMCStack.h"
#include "TMCManagerStack.h"
#include "TMCParticleStatus.h"

#include "TMCManager.h"

/** \class TMCManager
    \ingroup vmc

Singleton manager class for handling and steering a run with multiple TVirtualMC
engines sharing events.
It provides interfaces to transfer tracks between engines and exposes the correct
stacks to each running engine. A registered user stack is kept up-to-date
automatically seeing a consistent history.
Track objects (aka TParticle) are still owned by the user who must forward these to
the manager after creation. Everything else is done automatically.
*/

TMCThreadLocal TMCManager *TMCManager::fgInstance = nullptr;

////////////////////////////////////////////////////////////////////////////////
///
/// Default and standard constructor
///

TMCManager::TMCManager()
   : fApplication(nullptr), fCurrentEngine(nullptr), fTotalNPrimaries(0), fTotalNTracks(0), fUserStack(nullptr),
     fBranchArrayContainer(), fIsInitialized(kFALSE), fIsInitializedUser(kFALSE)
{
   if (fgInstance) {
      ::Fatal("TMCManager::TMCManager", "Attempt to create two instances of singleton.");
   }
   fgInstance = this;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Destructor
///

TMCManager::~TMCManager()
{
   for (auto &mc : fEngines) {
      delete mc;
   }
   fgInstance = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Static access method
///

TMCManager *TMCManager::Instance()
{
   return fgInstance;
}

////////////////////////////////////////////////////////////////////////////////
///
/// A TVirtualMC will register itself via this method during construction
/// if a TMCManager was instanciated before.
/// The TMCManager will assign an ID to the engines.
///

void TMCManager::Register(TVirtualMC *mc)
{
   // Do not register an engine twice.
   for (auto &currMC : fEngines) {
      if (currMC == mc) {
         ::Fatal("TMCManager::RegisterMC", "This engine is already registered.");
      }
   }
   // Set id and register.
   mc->SetId(fEngines.size());
   fEngines.push_back(mc);
   fStacks.emplace_back(new TMCManagerStack());
   mc->SetStack(fStacks.back().get());
   mc->SetManagerStack(fStacks.back().get());
   // Must update engine pointers here since during construction of the concrete TVirtualMC
   // implementation the static TVirtualMC::GetMC() or defined gMC might be used.
   UpdateEnginePointers(mc);
}

////////////////////////////////////////////////////////////////////////////////
///
/// The user application will register itself via this method when the
/// manager was requested.
///

void TMCManager::Register(TVirtualMCApplication *application)
{
   if (fApplication) {
      ::Fatal("TMCManager::Register", "The application is already registered.");
   }
   ::Info("TMCManager::Register", "Register user application and construct geometry");
   fApplication = application;
   // TODO Can these 3 functions can be called directly here? Or could any of these depend on an implemented VMC?
   fApplication->ConstructGeometry();
   fApplication->MisalignGeometry();
   fApplication->ConstructOpGeometry();
   if (!gGeoManager->IsClosed()) {
      // Setting the top volume is the duty of the user as well as closing it.
      // Failing here is just an additional cross check. If not closed the user
      // might have forgotten something.
      ::Fatal("TMCManager::Register", "The TGeo geometry is not closed. Please check whether you just have to close "
                                      "it or whether something was forgotten.");
   }
}

////////////////////////////////////////////////////////////////////////////////
///
/// Return the number of registered engines.
///

Int_t TMCManager::NEngines() const
{
   return fEngines.size();
}

////////////////////////////////////////////////////////////////////////////////
///
/// Get registered engine pointers
///

void TMCManager::GetEngines(std::vector<TVirtualMC *> &engines) const
{
   engines.clear();
   engines.resize(fEngines.size(), nullptr);
   for (UInt_t i = 0; i < fEngines.size(); i++) {
      engines[i] = fEngines[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///
/// Return the number of registered engines.
///

TVirtualMC *TMCManager::GetEngine(Int_t id) const
{
   if (id < 0 || id >= static_cast<Int_t>(fEngines.size())) {
      ::Fatal("TMCManager::GetEngine", "Unknown engine ID.");
   }
   return fEngines[id];
}

////////////////////////////////////////////////////////////////////////////////
///
/// Get engine ID by its name
///

Int_t TMCManager::GetEngineId(const char *engineName) const
{
   for (UInt_t i = 0; i < fEngines.size(); i++) {
      if (strcmp(engineName, fEngines[i]->GetName()) == 0) {
         return i;
      }
   }
   ::Warning("TMCManager::GetEngineId", "Unknown engine %s.", engineName);
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Get the current engine pointer
///

TVirtualMC *TMCManager::GetCurrentEngine() const
{
   return fCurrentEngine;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Connect a pointer which is updated whenever the engine is changed
///

void TMCManager::ConnectEnginePointer(TVirtualMC **mc)
{
   fConnectedEnginePointers.push_back(mc);
   if (fCurrentEngine) {
      *mc = fCurrentEngine;
   }
}

////////////////////////////////////////////////////////////////////////////////
///
/// Connect a pointer which is updated whenever the engine is changed
///

void TMCManager::ConnectEnginePointer(TVirtualMC *&mc)
{
   ConnectEnginePointer(&mc);
}

////////////////////////////////////////////////////////////////////////////////
///
/// Set user stack
///

void TMCManager::SetUserStack(TVirtualMCStack *stack)
{
   fUserStack = stack;
}

////////////////////////////////////////////////////////////////////////////////
///
/// User interface to forward particle to specifiic engine.
/// It is assumed that the TParticle is owned by the user. It will not be
/// modified by the TMCManager.
///

void TMCManager::ForwardTrack(Int_t toBeDone, Int_t trackId, Int_t parentId, TParticle *particle, Int_t engineId)
{
   if (engineId < 0 || engineId >= static_cast<Int_t>(fEngines.size())) {
      ::Fatal("TMCManager::ForwardTrack", "Engine ID %i out of bounds. Have %zu engines.", engineId, fEngines.size());
   }
   if (trackId >= static_cast<Int_t>(fParticles.size())) {
      fParticles.resize(trackId + 1, nullptr);
      fParticlesStatus.resize(trackId + 1);
   }
   fParticles[trackId] = particle;
   fParticlesStatus[trackId].reset(new TMCParticleStatus());
   fParticlesStatus[trackId]->fId = trackId;
   fParticlesStatus[trackId]->fParentId = parentId;
   fParticlesStatus[trackId]->InitFromParticle(particle);
   fTotalNTracks++;
   if (particle->IsPrimary()) {
      fTotalNPrimaries++;
   }

   if (toBeDone > 0) {
      if (particle->IsPrimary()) {
         fStacks[engineId]->PushPrimaryTrackId(trackId);
      } else {
         fStacks[engineId]->PushSecondaryTrackId(trackId);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
///
/// User interface to forward particle to specifiic engine.
/// It is assumed that the TParticle is owned by the user. It will not be
/// modified by the TMCManager.
/// Assume current engine Id
///

void TMCManager::ForwardTrack(Int_t toBeDone, Int_t trackId, Int_t parentId, TParticle *particle)
{
   ForwardTrack(toBeDone, trackId, parentId, particle, fCurrentEngine->GetId());
}

////////////////////////////////////////////////////////////////////////////////
///
/// Transfer track from current engine to engine with engineTargetId
///

void TMCManager::TransferTrack(Int_t engineTargetId)
{
   if (engineTargetId < 0 || engineTargetId >= static_cast<Int_t>(fEngines.size())) {
      ::Fatal("TMCManager::TransferTrack",
              "Target engine ID out of bounds. Have %zu engines. Requested target ID was %i", fEngines.size(),
              engineTargetId);
   }
   TransferTrack(fEngines[engineTargetId]);
}

////////////////////////////////////////////////////////////////////////////////
///
/// Transfer track from current engine to target engine mc
///

void TMCManager::TransferTrack(TVirtualMC *mc)
{
   // Do nothing if target and current engines are the same
   if (mc == fCurrentEngine) {
      return;
   }

   // Get information on current track and extract status from transporting engine
   Int_t trackId = fStacks[fCurrentEngine->GetId()]->GetCurrentTrackNumber();

   fCurrentEngine->TrackPosition(fParticlesStatus[trackId]->fPosition);
   fCurrentEngine->TrackMomentum(fParticlesStatus[trackId]->fMomentum);
   fCurrentEngine->TrackPolarization(fParticlesStatus[trackId]->fPolarization);
   fParticlesStatus[trackId]->fStepNumber = fCurrentEngine->StepNumber();
   fParticlesStatus[trackId]->fTrackLength = fCurrentEngine->TrackLength();
   fParticlesStatus[trackId]->fWeight = fCurrentEngine->TrackWeight();

   // Store TGeoNavidator's fIsOutside state
   fParticlesStatus[trackId]->fIsOutside = gGeoManager->IsOutside();

   TGeoBranchArray *geoState = fBranchArrayContainer.GetNewGeoState(fParticlesStatus[trackId]->fGeoStateIndex);
   geoState->InitFromNavigator(gGeoManager->GetCurrentNavigator());

   // Push only the particle ID
   if (fParticles[trackId]->IsPrimary()) {
      fStacks[mc->GetId()]->PushPrimaryTrackId(trackId);
   } else {
      fStacks[mc->GetId()]->PushSecondaryTrackId(trackId);
   }
   fCurrentEngine->InterruptTrack();
}

////////////////////////////////////////////////////////////////////////////////
///
/// Try to restore geometry for a given track
///

Bool_t TMCManager::RestoreGeometryState(Int_t trackId, Bool_t checkTrackIdRange)
{
  if (checkTrackIdRange && (trackId < 0 || trackId >= static_cast<Int_t>(fParticles.size()) || !fParticles[trackId])) {
     return kFALSE;
  }
  UInt_t& geoStateId = fParticlesStatus[trackId]->fGeoStateIndex;
  if(geoStateId == 0) {
    return kFALSE;
  }
  const TGeoBranchArray* branchArray = fBranchArrayContainer.GetGeoState(geoStateId);
  branchArray->UpdateNavigator(gGeoManager->GetCurrentNavigator());
  fBranchArrayContainer.FreeGeoState(geoStateId);
  gGeoManager->SetOutside(fParticlesStatus[trackId]->fIsOutside);
  geoStateId = 0;
  return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Try to restore geometry for the track currently set
///

Bool_t TMCManager::RestoreGeometryState()
{
  return RestoreGeometryState(fStacks[fCurrentEngine->GetId()]->GetCurrentTrackNumber(), kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
///
/// Initialize engines
///

void TMCManager::Init()
{
   if (fIsInitialized) {
      return;
   }

   if (!fUserStack) {
      ::Fatal("TMCManager::Run", "Missing user stack pointer.");
   }
   if (fEngines.empty()) {
      ::Fatal("TMCManager::Run", "No engines registered");
   }

   for (auto &mc : fEngines) {
      // Must have geometry handling via TGeo
      if (!mc->IsRootGeometrySupported()) {
         ::Fatal("TMCManager::Run", "Engine %s does not support geometry built via ROOT's TGeoManager", mc->GetName());
      }
      Int_t currentEngineId = mc->GetId();
      // To be able to forward info to the user stack
      fStacks[currentEngineId]->SetUserStack(fUserStack);
      // Connect the engine's stack to the centrally managed vectors
      fStacks[currentEngineId]->ConnectTrackContainers(&fParticles, &fParticlesStatus, &fBranchArrayContainer,
                                                       &fTotalNPrimaries, &fTotalNTracks);
   }

   // Initialize the fBranchArrayContainer to manage and cache TGeoBranchArrays
   fBranchArrayContainer.InitializeFromGeoManager(gGeoManager);

   fIsInitialized = kTRUE;

   // Send warning if only one engine ==> overhead
   if (fEngines.size() == 1) {
      ::Warning("TMCManager::Run", "Only one engine registered. That will lead to overhead in "
                                   "the simulation run due to additional hooks and dispatches "
                                   "to the TMCManager.");
   }
}

////////////////////////////////////////////////////////////////////////////////
///
/// Run the event loop
///

void TMCManager::Run(Int_t nEvents)
{
   if (!fIsInitialized) {
      ::Fatal("TMCManager::Run", "Engines have not yet been initialized.");
   }

   // Set user initialize true in any case now so that this cannot happen by accident during a run
   fIsInitializedUser = kTRUE;

   if (nEvents < 1) {
      ::Fatal("TMCManager::Run", "Need at least one event to process but %i events specified.", nEvents);
   }

   // Run 1 event nEvents times
   for (Int_t i = 0; i < nEvents; i++) {
      ::Info("TMCManager::Run", "Start event %i", i + 1);
      PrepareNewEvent();
      fApplication->BeginEvent();
      // Loop as long as there are tracks in any engine stack
      while (GetNextEngine()) {
         fCurrentEngine->ProcessEvent(i, kTRUE);
      }
      fApplication->FinishEvent();
   }
   TerminateRun();
}

////////////////////////////////////////////////////////////////////////////////
///
/// Choose next engines to be run in the loop
///

void TMCManager::PrepareNewEvent()
{
   fBranchArrayContainer.FreeGeoStates();
   // Reset in event flag for all engines and clear stacks
   for (auto &stack : fStacks) {
      stack->ResetInternals();
   }
   for (UInt_t i = 0; i < fParticles.size(); i++) {
      fParticlesStatus.clear();
      fParticlesStatus.resize(fParticles.size());
      fParticles[i] = nullptr;
   }

   // GeneratePrimaries centrally
   fApplication->GeneratePrimaries();
}

////////////////////////////////////////////////////////////////////////////////
///
/// Choose next engines to be run in the loop
///

Bool_t TMCManager::GetNextEngine()
{
   // Select next engine based on finite number of particles on the stack
   for (UInt_t i = 0; i < fStacks.size(); i++) {
      if (fStacks[i]->GetStackedNtrack() > 0) {
         UpdateEnginePointers(fEngines[i]);
         return kTRUE;
      }
   }
   // No tracks to be processed.
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Update all engine pointers connected to the TMCManager
///

void TMCManager::UpdateEnginePointers(TVirtualMC *mc)
{
   fCurrentEngine = mc;
   for (TVirtualMC **&mcPtr : fConnectedEnginePointers) {
      *mcPtr = mc;
   }
   // Make sure TVirtualMC::GetMC() returns the current engine.
   TVirtualMC::fgMC = mc;
}

////////////////////////////////////////////////////////////////////////////////
///
/// Terminate the run for all engines
///

void TMCManager::TerminateRun()
{
   for (auto &mc : fEngines) {
      mc->TerminateRun();
   }
}
