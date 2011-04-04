// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveVSDStructs
#define ROOT_TEveVSDStructs

#include "TObject.h"
#include "TParticle.h"
#include "TEveVector.h"

/******************************************************************************/
// VSD Structures
/******************************************************************************/

// Basic structures for Reve VSD concept. Design criteria:
//
//  * provide basic cross-referencing functionality;
//
//  * small memory/disk footprint (floats / count on compression in
//    split mode);
//
//  * simple usage from tree selections;
//
//  * placement in TClonesArray (composites are TObject derived);
//
//  * minimal member-naming (impossible to make everybody happy).
//


/******************************************************************************/
// TEveMCTrack
/******************************************************************************/

class TEveMCTrack : public TParticle // ?? Copy stuff over ??
{
public:
   Int_t       fLabel;      // Label of the track
   Int_t       fIndex;      // Index of the track (in some source array)
   Int_t       fEvaLabel;   // Label of primary particle

   Bool_t      fDecayed;    // True if decayed during tracking.
   // ?? Perhaps end-of-tracking point/momentum would be better.
   Float_t     fTDecay;     // Decay time
   TEveVector  fVDecay;     // Decay vertex
   TEveVector  fPDecay;     // Decay momentum

   TEveMCTrack() : fLabel(-1), fIndex(-1), fEvaLabel(-1),
                   fDecayed(kFALSE), fTDecay(0), fVDecay(), fPDecay() {}
   virtual ~TEveMCTrack() {}

   TEveMCTrack& operator=(const TParticle& p)
   { *((TParticle*)this) = p; return *this; }

   void ResetPdgCode() { fPdgCode = 0; }

   ClassDef(TEveMCTrack, 1); // Monte Carlo track (also used in VSD).
};


/******************************************************************************/
// TEveHit
/******************************************************************************/

// Representation of a hit.

// Members det_id (and fSubdetId) serve for cross-referencing into
// geometry. Hits should be stored in fDetId (+some label ordering) in
// order to maximize branch compression.


class TEveHit : public TObject
{
public:
   UShort_t     fDetId;    // Custom detector id.
   UShort_t     fSubdetId; // Custom sub-detector id.
   Int_t        fLabel;    // Label of particle that produced the hit.
   Int_t        fEvaLabel; // Label of primary particle, ancestor of label.
   TEveVector   fV;        // Hit position.

   // Float_t charge; probably specific.

   TEveHit() : fDetId(0), fSubdetId(0), fLabel(0), fEvaLabel(0), fV() {}
   virtual ~TEveHit() {}

   ClassDef(TEveHit, 1); // Monte Carlo hit (also used in VSD).
};


/******************************************************************************/
// TEveCluster
/******************************************************************************/

// Base class for reconstructed clusters

// ?? Should TEveHit and cluster have common base? No.

class TEveCluster : public TObject
{
public:
   UShort_t     fDetId;     // Custom detector id.
   UShort_t     fSubdetId;  // Custom sub-detector id.
   Int_t        fLabel[3];  // Labels of particles that contributed hits.

   // ?? Should include reconstructed track(s) using it? Rather not, separate.

   TEveVector      fV;      // Vertex.
   // TEveVector   fW;      // Cluster widths.
   // Coord system? Errors and/or widths Wz, Wy?

   TEveCluster() : fDetId(0), fSubdetId(0), fV() { fLabel[0] = fLabel[1] = fLabel[2] = 0; }
   virtual ~TEveCluster() {}

   ClassDef(TEveCluster, 1); // Reconstructed cluster (also used in VSD).
};


/******************************************************************************/
// TEveRecTrack
/******************************************************************************/
template <typename TT>
class TEveRecTrackT : public TObject
{
public:
   Int_t           fLabel;       // Label of the track.
   Int_t           fIndex;       // Index of the track (in some source array).
   Int_t           fStatus;      // Status as exported from reconstruction.
   Int_t           fSign;        // Charge of the track.
   TEveVectorT<TT> fV;           // Start vertex from reconstruction.
   TEveVectorT<TT> fP;           // Reconstructed momentum at start vertex.
   TT              fBeta;        // Relativistic beta factor.

   // PID data missing

   TEveRecTrackT() : fLabel(-1), fIndex(-1), fStatus(0), fSign(0), fV(), fP(), fBeta(0) {}
   virtual ~TEveRecTrackT() {}

   Float_t Pt() { return fP.Perp(); }

   ClassDef(TEveRecTrackT, 1); // Template for reconstructed track (also used in VSD).
};

typedef TEveRecTrackT<Float_t>  TEveRecTrack;
typedef TEveRecTrackT<Float_t>  TEveRecTrackF;
typedef TEveRecTrackT<Double_t> TEveRecTrackD;

/******************************************************************************/
// TEveRecKink
/******************************************************************************/

class TEveRecKink : public TObject
{
public:

   TEveVector  fVKink;          // Kink vertex: reconstructed position of the kink
   TEveVector  fPMother;        // Momentum of the mother track
   TEveVector  fVMother;        // Vertex of the mother track
   TEveVector  fPDaughter;      // Momentum of the daughter track
   TEveVector  fVDaughter;      // Vertex of the daughter track
   Double32_t  fKinkAngle[3];   // three angles
   Int_t       fSign;           // sign of the track
   Int_t       fStatus;         // Status as exported from reconstruction

   // Data from simulation
   Int_t       fKinkLabel[2];   // Labels of the mother and daughter tracks
   Int_t       fKinkIndex[2];   // Indices of the mother and daughter tracks
   Int_t       fKinkPdg[2];     // PDG code of mother and daughter.

   TEveRecKink() : fVKink(), fPMother(), fVMother(), fPDaughter(), fVDaughter(), fSign(0), fStatus(0)
   {
     fKinkAngle[0] = fKinkAngle[1] = fKinkAngle[2] = 0;
     fKinkLabel[0] = fKinkLabel[1] = 0;
     fKinkIndex[0] = fKinkIndex[1] = 0;
     fKinkPdg[0]   = fKinkPdg[1]   = 0;
   }
   virtual ~TEveRecKink() {}

   ClassDef(TEveRecKink, 1); // Reconstructed kink (also used in VSD).
};


/******************************************************************************/
// TEveRecV0
/******************************************************************************/

class TEveRecV0 : public TObject
{
public:
   Int_t      fStatus;

   TEveVector fVNeg;       // Vertex of negative track.
   TEveVector fPNeg;       // Momentum of negative track.
   TEveVector fVPos;       // Vertex of positive track.
   TEveVector fPPos;       // Momentum of positive track.

   TEveVector fVCa;        // Point of closest approach.
   TEveVector fV0Birth;    // Reconstucted birth point of neutral particle.

   // ? Data from simulation.
   Int_t      fLabel;      // Neutral mother label read from kinematics.
   Int_t      fPdg;        // PDG code of mother.
   Int_t      fDLabel[2];  // Daughter labels.

   TEveRecV0() : fStatus(), fVNeg(), fPNeg(), fVPos(), fPPos(),
                 fVCa(), fV0Birth(), fLabel(0), fPdg(0)
   { fDLabel[0] = fDLabel[1] = 0; }
   virtual ~TEveRecV0() {}

   ClassDef(TEveRecV0, 1); // Reconstructed V0 (also used in VSD).
};


/******************************************************************************/
// TEveRecCascade
/******************************************************************************/

class TEveRecCascade : public TObject
{
public:
   Int_t      fStatus;

   TEveVector fVBac;         // Vertex of bachelor track.
   TEveVector fPBac;         // Momentum of bachelor track.

   TEveVector fCascadeVCa;   // Point of closest approach for Cascade.
   TEveVector fCascadeBirth; // Reconstucted birth point of cascade particle.

   // ? Data from simulation.
   Int_t      fLabel;        // Cascade mother label read from kinematics.
   Int_t      fPdg;          // PDG code of mother.
   Int_t      fDLabel;       // Daughter label.

   TEveRecCascade() : fStatus(),  fVBac(), fPBac(),
                      fCascadeVCa(), fCascadeBirth(),
                      fLabel(0), fPdg(0), fDLabel(0) {}
   virtual ~TEveRecCascade() {}

   ClassDef(TEveRecCascade, 1); // Reconstructed Cascade (also used in VSD).
};


/******************************************************************************/
// TEveMCRecCrossRef
/******************************************************************************/

class TEveMCRecCrossRef : public TObject
{
public:
   Bool_t       fIsRec;   // Is reconstructed.
   Bool_t       fHasV0;
   Bool_t       fHasKink;
   Int_t        fLabel;
   Int_t        fNHits;
   Int_t        fNClus;

   TEveMCRecCrossRef() : fIsRec(false), fHasV0(false), fHasKink(false),
                         fLabel(0), fNHits(0), fNClus(0) {}
   virtual ~TEveMCRecCrossRef() {}

   ClassDef(TEveMCRecCrossRef, 1); // Cross-reference of sim/rec data per particle (also used in VSD).
};


/******************************************************************************/
// Missing primary vertex class.
/******************************************************************************/


/******************************************************************************/
/******************************************************************************/

// This whole construction is somewhat doubtable. It requires
// shameless copying of experiment data. What is good about this
// scheme:
//
// 1) Filters can be applied at copy time so that only part of the
// data is copied over.
//
// 2) Once the data is extracted it can be used without experiment
// software. Thus, external service can provide this data and local
// client can be really thin.
//
// 3) Some pretty advanced visualization schemes/selections can be
// implemented in a general framework by providing data extractors
// only. This is also good for PR or VIP displays.
//
// 4) These classes can be extended by particular implementations. The
// container classes will use TClonesArray with user-specified element
// class.

// The common behaviour could be implemented entirely without usage of
// a common base classes, by just specifying names of members that
// retrieve specific data. This is fine as long as one only uses tree
// selections but becomes painful for extraction of data into local
// structures (could a) use interpreter but this is an overkill and
// would cause serious trouble for multi-threaded environment; b) use
// member offsets and data-types from the dictionary).

#endif
