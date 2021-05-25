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

#ifndef ROOT_TMCManagerStack
#define ROOT_TMCManagerStack

// Class TMCManagerStack
// ---------------------
// stack used by the TMCManager when handling multiple engines
//

#include <vector>
#include <stack>
#include <memory>

#include "TMCtls.h"
#include "TLorentzVector.h"
#include "TMCProcess.h"

#include "TVirtualMCStack.h"

struct TMCParticleStatus;
class TGeoBranchArray;
class TGeoMCBranchArrayContainer;

class TMCManagerStack : public TVirtualMCStack {

public:
   /// Default constructor
   TMCManagerStack();
   /// Destructor
   virtual ~TMCManagerStack() = default;

   //
   // Methods for stacking
   //

   /// This will just forward the call to the fUserStack's PushTrack
   ///
   /// Create a new particle and push into stack;
   /// - toBeDone   - 1 if particles should go to tracking, 0 otherwise
   /// - parent     - number of the parent track, -1 if track is primary
   /// - pdg        - PDG encoding
   /// - px, py, pz - particle momentum [GeV/c]
   /// - e          - total energy [GeV]
   /// - vx, vy, vz - position [cm]
   /// - tof        - time of flight [s]
   /// - polx, poly, polz - polarization
   /// - mech       - creator process VMC code
   /// - ntr        - track number (is filled by the stack
   /// - weight     - particle weight
   /// - is         - generation status code
   void PushTrack(Int_t toBeDone, Int_t parent, Int_t pdg, Double_t px, Double_t py, Double_t pz, Double_t e,
                  Double_t vx, Double_t vy, Double_t vz, Double_t tof, Double_t polx, Double_t poly, Double_t polz,
                  TMCProcess mech, Int_t &ntr, Double_t weight, Int_t is) override final;

   //
   // Get methods
   //

   /// Pop next track
   TParticle *PopNextTrack(Int_t &itrack) override final;

   /// Pop i'th primar, that does not mean that this primariy also has ID==i
   TParticle *PopPrimaryForTracking(Int_t i) override final;

   /// Pop i'th primary, that does not mean that this primariy also has ID==i.
   /// including actual index
   TParticle *PopPrimaryForTracking(Int_t i, Int_t &itrack);

   /// Get number of tracks on current sub-stack
   Int_t GetNtrack() const override final;

   /// Get only the number of currently stacked tracks
   Int_t GetStackedNtrack() const;

   /// Get number of primaries on current sub-stack
   Int_t GetNprimary() const override final;

   /// Get only the number of currently stacked primaries
   Int_t GetStackedNprimary() const;

   /// Current track
   TParticle *GetCurrentTrack() const override final;

   /// Current track number
   Int_t GetCurrentTrackNumber() const override final;

   /// Number of the parent of the current track
   Int_t GetCurrentParentTrackNumber() const override final;

   /// Set the current track id from the outside and forward this to the
   /// user's stack
   void SetCurrentTrack(Int_t trackId) override final;

   /// Get TMCParticleStatus by trackId
   const TMCParticleStatus *GetParticleStatus(Int_t trackId) const;

   /// Get particle's geometry status by trackId
   const TGeoBranchArray *GetGeoState(Int_t trackId) const;

   /// Get current particle's geometry status
   const TGeoBranchArray *GetCurrentGeoState() const;

private:
   friend class TMCManager;
   /// Check whether track trackId exists
   Bool_t HasTrackId(Int_t trackId) const;
   /// Set the user stack
   void SetUserStack(TVirtualMCStack *stack);
   /// Set the pointer to vector with all particles and status
   void ConnectTrackContainers(std::vector<TParticle *> *particles,
                               std::vector<std::unique_ptr<TMCParticleStatus>> *tracksStatus,
                               TGeoMCBranchArrayContainer *branchArrayContainer, Int_t *totalNPrimaries,
                               Int_t *totalNTracks);
   /// Push primary id to be processed
   void PushPrimaryTrackId(Int_t trackId);
   /// Push secondary id to be processed
   void PushSecondaryTrackId(Int_t trackId);
   /// Reset internals, clear engine stack and fParticles and reset buffered values
   void ResetInternals();

private:
   /// Pointer to current track
   Int_t fCurrentTrackId;
   /// Pointer to user stack for forwarding PushTrack calls
   TVirtualMCStack *fUserStack;
   /// Number of all primaries ever pushed linked from the TMCManager
   Int_t *fTotalNPrimaries;
   /// Number of all tracks ever pushed linked from the TMCManager
   Int_t *fTotalNTracks;
   /// All tracks linked from the TMCManager
   std::vector<TParticle *> *fParticles;
   /// All TMCParticleStatus linked from the TMCManager
   std::vector<std::unique_ptr<TMCParticleStatus>> *fParticlesStatus;
   /// Storage of TGeoBranchArray pointers
   TGeoMCBranchArrayContainer *fBranchArrayContainer;
   /// IDs of primaries to be tracked
   std::stack<Int_t> fPrimariesStack;
   /// IDs of secondaries to be trackedk
   std::stack<Int_t> fSecondariesStack;

   ClassDefOverride(TMCManagerStack, 1)
};

#endif // ROOT_TMCManagerStack
