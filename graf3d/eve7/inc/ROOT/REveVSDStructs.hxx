// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveVSDStructs
#define ROOT7_REveVSDStructs

#include "TParticle.h"

#include <ROOT/REveVector.hxx>

////////////////////////////////////////////////////////////////////////////////
/// VSD Structures
////////////////////////////////////////////////////////////////////////////////

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

namespace ROOT {
namespace Experimental {

////////////////////////////////////////////////////////////////////////////////
/// REveMCTrack
////////////////////////////////////////////////////////////////////////////////

class REveMCTrack : public TParticle // ?? Copy stuff over ??
{
public:
   Int_t fLabel{-1};    // Label of the track
   Int_t fIndex{-1};    // Index of the track (in some source array)
   Int_t fEvaLabel{-1}; // Label of primary particle

   Bool_t fDecayed{kFALSE}; // True if decayed during tracking.
   // ?? Perhaps end-of-tracking point/momentum would be better.
   Float_t fTDecay{0};    // Decay time
   REveVector fVDecay; // Decay vertex
   REveVector fPDecay; // Decay momentum

   REveMCTrack() = default;
   virtual ~REveMCTrack() {}

   REveMCTrack &operator=(const TParticle &p)
   {
      *((TParticle *)this) = p;
      return *this;
   }

   void ResetPdgCode() { fPdgCode = 0; }

   ClassDef(REveMCTrack, 1); // Monte Carlo track (also used in VSD).
};

////////////////////////////////////////////////////////////////////////////////
/// REveHit
/// Monte Carlo hit (also used in VSD).
///
/// Representation of a hit.
/// Members det_id (and fSubdetId) serve for cross-referencing into
/// geometry. Hits should be stored in fDetId (+some label ordering) in
/// order to maximize branch compression.
////////////////////////////////////////////////////////////////////////////////


class REveHit
{
public:
   UShort_t fDetId{0};    // Custom detector id.
   UShort_t fSubdetId{0}; // Custom sub-detector id.
   Int_t fLabel{0};       // Label of particle that produced the hit.
   Int_t fEvaLabel{0};    // Label of primary particle, ancestor of label.
   REveVector fV;      // Hit position.

   // Float_t charge; probably specific.

   REveHit() = default;
   virtual ~REveHit() {}
};

////////////////////////////////////////////////////////////////////////////////
/// REveCluster
/// Reconstructed cluster (also used in VSD).
////////////////////////////////////////////////////////////////////////////////

// Base class for reconstructed clusters

// ?? Should REveHit and cluster have common base? No.

class REveCluster // : public TObject
{
public:
   UShort_t fDetId{0};    // Custom detector id.
   UShort_t fSubdetId{0}; // Custom sub-detector id.
   Int_t fLabel[3];    // Labels of particles that contributed hits.

   // ?? Should include reconstructed track(s) using it? Rather not, separate.

   REveVector fV; // Vertex.
   // REveVector   fW;      // Cluster widths.
   // Coord system? Errors and/or widths Wz, Wy?

   REveCluster() { fLabel[0] = fLabel[1] = fLabel[2] = 0; }
   virtual ~REveCluster() {}
};

////////////////////////////////////////////////////////////////////////////////
/// REveRecTrack
/// Template for reconstructed track (also used in VSD).
////////////////////////////////////////////////////////////////////////////////

template <typename TT>
class REveRecTrackT
{
public:
   Int_t fLabel{-1};       // Label of the track.
   Int_t fIndex{-1};       // Index of the track (in some source array).
   Int_t fStatus{0};      // Status as exported from reconstruction.
   Int_t fSign{0};        // Charge of the track.
   REveVectorT<TT> fV; // Start vertex from reconstruction.
   REveVectorT<TT> fP; // Reconstructed momentum at start vertex.
   TT fBeta{0};           // Relativistic beta factor.
   Double32_t fDcaXY{0};  // dca xy to the primary vertex
   Double32_t fDcaZ{0};   // dca z to the primary vertex
   Double32_t fPVX{0};    //
   Double32_t fPVY{0};    //
   Double32_t fPVZ{0};    //
   // PID data missing

   REveRecTrackT() = default;
   virtual ~REveRecTrackT() {}

   Float_t Pt() { return fP.Perp(); }
};

typedef REveRecTrackT<Float_t> REveRecTrack;
typedef REveRecTrackT<Float_t> REveRecTrackF;
typedef REveRecTrackT<Double_t> REveRecTrackD;

////////////////////////////////////////////////////////////////////////////////
/// REveRecKink
/// Reconstructed kink (also used in VSD).
////////////////////////////////////////////////////////////////////////////////

class REveRecKink // : public TObject
{
public:
   REveVector fVKink;        // Kink vertex: reconstructed position of the kink
   REveVector fPMother;      // Momentum of the mother track
   REveVector fVMother;      // Vertex of the mother track
   REveVector fPDaughter;    // Momentum of the daughter track
   REveVector fVDaughter;    // Vertex of the daughter track
   Double32_t fKinkAngle[3]; // three angles
   Int_t fSign{0};           // sign of the track
   Int_t fStatus{0};         // Status as exported from reconstruction

   // Data from simulation
   Int_t fKinkLabel[2]; // Labels of the mother and daughter tracks
   Int_t fKinkIndex[2]; // Indices of the mother and daughter tracks
   Int_t fKinkPdg[2];   // PDG code of mother and daughter.

   REveRecKink()
   {
      fKinkAngle[0] = fKinkAngle[1] = fKinkAngle[2] = 0;
      fKinkLabel[0] = fKinkLabel[1] = 0;
      fKinkIndex[0] = fKinkIndex[1] = 0;
      fKinkPdg[0] = fKinkPdg[1] = 0;
   }
   virtual ~REveRecKink() {}
};

////////////////////////////////////////////////////////////////////////////////
/// REveRecV0
////////////////////////////////////////////////////////////////////////////////

class REveRecV0
{
public:
   Int_t fStatus{0};

   REveVector fVNeg; // Vertex of negative track.
   REveVector fPNeg; // Momentum of negative track.
   REveVector fVPos; // Vertex of positive track.
   REveVector fPPos; // Momentum of positive track.

   REveVector fVCa;     // Point of closest approach.
   REveVector fV0Birth; // Reconstructed birth point of neutral particle.

   // ? Data from simulation.
   Int_t fLabel{0};     // Neutral mother label read from kinematics.
   Int_t fPdg{0};       // PDG code of mother.
   Int_t fDLabel[2];    // Daughter labels.

   REveRecV0() { fDLabel[0] = fDLabel[1] = 0; }
   virtual ~REveRecV0() {}
};

////////////////////////////////////////////////////////////////////////////////
/// REveRecCascade
////////////////////////////////////////////////////////////////////////////////

class REveRecCascade
{
public:
   Int_t fStatus{0};

   REveVector fVBac; // Vertex of bachelor track.
   REveVector fPBac; // Momentum of bachelor track.

   REveVector fCascadeVCa;   // Point of closest approach for Cascade.
   REveVector fCascadeBirth; // Reconstructed birth point of cascade particle.

   // ? Data from simulation.
   Int_t fLabel{0};  // Cascade mother label read from kinematics.
   Int_t fPdg{0};    // PDG code of mother.
   Int_t fDLabel{0}; // Daughter label.

   REveRecCascade() = default;
   virtual ~REveRecCascade() {}
};

////////////////////////////////////////////////////////////////////////////////
/// REveMCRecCrossRef
/// Cross-reference of sim/rec data per particle (also used in VSD).
////////////////////////////////////////////////////////////////////////////////

class REveMCRecCrossRef {
public:
   Bool_t fIsRec{kFALSE}; // Is reconstructed.
   Bool_t fHasV0{kFALSE};
   Bool_t fHasKink{kFALSE};
   Int_t fLabel{0};
   Int_t fNHits{0};
   Int_t fNClus{0};

   REveMCRecCrossRef() = default;
   virtual ~REveMCRecCrossRef() {}
};

////////////////////////////////////////////////////////////////////////////////
/// Missing primary vertex class.
////////////////////////////////////////////////////////////////////////////////

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

} // namespace Experimental
} // namespace ROOT

#endif
