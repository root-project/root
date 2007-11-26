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

#include <TObject.h>
#include <TMath.h>

#include <TParticle.h>

/******************************************************************************/
// VSD Structures
/******************************************************************************/

// Basic structures for Reve VSD concept. Design criteria:
//
//  * provide basic cross-referencing functionality;
//
//  * small memory/disk footprint (floats / count on compression in
//  split mode);
//
//  * simple usage from tree selections;
//
//  * placement in TClonesArray (composites are TObject derived);
//
//  * minimal member-naming (impossible to make everybody happy).


/******************************************************************************/
// TEveVector
/******************************************************************************/

class TEveVector
{
public:
   Float_t x, y, z;

   TEveVector() : x(0), y(0), z(0) {}
   TEveVector(Float_t _x, Float_t _y, Float_t _z) : x(_x), y(_y), z(_z) {}
   virtual ~TEveVector() {}

   TEveVector operator + (const TEveVector &);
   TEveVector operator - (const TEveVector &);
   TEveVector operator * (Float_t a);

   Float_t* c_vec() { return &x; }
   Float_t& operator [] (Int_t indx);
   Float_t  operator [] (Int_t indx) const;

   void Set(Float_t*  v) { x=v[0]; y=v[1]; z=v[2]; }
   void Set(Double_t* v) { x=v[0]; y=v[1]; z=v[2]; }
   void Set(Float_t  _x, Float_t  _y, Float_t  _z) { x=_x; y=_y; z=_z; }
   void Set(Double_t _x, Double_t _y, Double_t _z) { x=_x; y=_y; z=_z; }
   void Set(const TVector3& v) { x=v.x(); y=v.y(); z=v.z(); }
   void Set(const TEveVector& v) { x=v.x; y=v.y; z=v.z; }

   Float_t Phi()      const;
   Float_t Theta()    const;
   Float_t CosTheta() const;
   Float_t Eta()      const;

   Float_t Mag()  const { return TMath::Sqrt(x*x+y*y+z*z);}
   Float_t Mag2() const { return x*x+y*y+z*z;}

   Float_t Perp()  const { return TMath::Sqrt(x*x+y*y);}
   Float_t Perp2() const { return x*x+y*y;}
   Float_t R()     const { return Perp(); }

   Float_t Distance(const TEveVector& v) const;
   Float_t SquareDistance(const TEveVector& v) const;
   Float_t Dot(const TEveVector&a) const;

   TEveVector& Mult(const TEveVector&a, Float_t af) { x = a.x*af; y = a.y*af; z = a.z*af; return *this; }


   ClassDef(TEveVector, 1); // Float three-vector; a inimal Float_t copy of TVector3 used to represent points and momenta (also used in VSD).
};

//______________________________________________________________________________
inline Float_t TEveVector::Phi() const
{ return x == 0.0 && y == 0.0 ? 0.0 : TMath::ATan2(y,x); }

inline Float_t TEveVector::Theta() const
{ return x == 0.0 && y == 0.0 && z == 0.0 ? 0.0 : TMath::ATan2(Perp(),z); }

inline Float_t TEveVector::CosTheta() const
{ Float_t ptot = Mag(); return ptot == 0.0 ? 1.0 : z/ptot; }

inline Float_t TEveVector::Distance( const TEveVector& b) const
{
   return TMath::Sqrt((x - b.x)*(x - b.x) + (y - b.y)*(y - b.y) + (z - b.z)*(z - b.z));
}
inline Float_t TEveVector::SquareDistance(const TEveVector& b) const
{
   return ((x - b.x)*(x - b.x) + (y - b.y)*(y - b.y) + (z - b.z)*(z - b.z));
}

//______________________________________________________________________________
inline Float_t TEveVector::Dot(const TEveVector& a) const
{
   return a.x*x + a.y*y + a.z*z;
}

inline Float_t& TEveVector::operator [] (Int_t idx)
{ return (&x)[idx]; }

inline Float_t TEveVector::operator [] (Int_t idx) const
{ return (&x)[idx]; }


/******************************************************************************/
// TEvePathMark
/******************************************************************************/

class TEvePathMark
{
public:
   enum Type_e   { Reference, Daughter, Decay };

   TEveVector  V;    // vertex
   TEveVector  P;    // momentum
   Float_t     time; // time
   Type_e      type; // mark-type

   TEvePathMark(Type_e t=Reference) : V(), P(), time(0), type(t) {}
   virtual ~TEvePathMark() {}

   const char* type_name();

   ClassDef(TEvePathMark, 1); // Special-point on track: position/momentum reference, daughter creation or decay (also used in VSD).
};

/******************************************************************************/
// TEveMCTrack
/******************************************************************************/

class TEveMCTrack : public TParticle // ?? Copy stuff over ??
{
public:
   Int_t       label;       // Label of the track
   Int_t       index;       // Index of the track (in some source array)
   Int_t       eva_label;   // Label of primary particle

   Bool_t      decayed;     // True if decayed during tracking.
   // ?? Perhaps end-of-tracking point/momentum would be better.
   Float_t     t_decay;     // Decay time
   TEveVector  V_decay;     // Decay vertex
   TEveVector  P_decay;     // Decay momentum

   TEveMCTrack() : label(-1), index(-1), eva_label(-1),
                   decayed(false), t_decay(0), V_decay(), P_decay() {}
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

// Members det_id (and subdet_id) serve for cross-referencing into
// geometry. Hits should be stored in det_id (+some label ordering) in
// order to maximize branch compression.


class TEveHit : public TObject
{
public:
   UShort_t     det_id;    // Custom detector id
   UShort_t     subdet_id; // Custom sub-detector id
   Int_t        label;     // Label of particle that produced the hit
   Int_t        eva_label; // Label of primary particle, ancestor of label
   TEveVector   V;     // Hit position

   // Float_t charge; Probably specific.

   TEveHit() : det_id(0), subdet_id(0), label(0), eva_label(0), V() {}
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
   UShort_t     det_id;    // Custom detector id
   UShort_t     subdet_id; // Custom sub-detector id
   Int_t        label[3];  // Labels of particles that contributed hits
   // ?? Should include reconstructed track using it? Rather not, separate.

   TEveVector   V;         // Vertex
   // TEveVector   W;      // Cluster widths
   // ?? Coord system? Special variables Wz, Wy?

   TEveCluster() : det_id(0), subdet_id(0), V() { label[0] = label[1] = label [2] = 0; }
   virtual ~TEveCluster() {}

   ClassDef(TEveCluster, 1); // Reconstructed cluster (also used in VSD).
};


/******************************************************************************/
// TEveRecTrack
/******************************************************************************/

class TEveRecTrack : public TObject
{
public:
   Int_t       label;       // Label of the track
   Int_t       index;       // Index of the track (in some source array)
   Int_t       status;      // Status as exported from reconstruction
   Int_t       sign;        // Charge of the track
   TEveVector  V;           // Start vertex from reconstruction
   TEveVector  P;           // Reconstructed momentum at start vertex
   Float_t     beta;

   // PID data missing

   TEveRecTrack() : label(-1), index(-1), status(0), sign(0), V(), P(), beta(0) {}
   virtual ~TEveRecTrack() {}

   Float_t Pt() { return P.Perp(); }

   ClassDef(TEveRecTrack, 1); // Reconstructed track (also used in VSD).
};


/******************************************************************************/
// TEveRecKink
/******************************************************************************/

class TEveRecKink : public TEveRecTrack
{
public:
   Int_t       label_sec;  // Label of the secondary track
   TEveVector  V_end;      // End vertex: last point on the primary track
   TEveVector  V_kink;     // Kink vertex: reconstructed position of the kink
   TEveVector  P_sec;      // Momentum of secondary track

   TEveRecKink() : TEveRecTrack(), label_sec(0), V_end(), V_kink(), P_sec() {}
   virtual ~TEveRecKink() {}

   ClassDef(TEveRecKink, 1); // Reconstructed kink (also used in VSD).
};


/******************************************************************************/
// TEveRecV0
/******************************************************************************/

class TEveRecV0 : public TObject
{
public:
   Int_t      status;

   TEveVector V_neg;       // Vertex of negative track
   TEveVector P_neg;       // Momentum of negative track
   TEveVector V_pos;       // Vertex of positive track
   TEveVector P_pos;       // Momentum of positive track

   TEveVector V_ca;        // Point of closest approach
   TEveVector V0_birth;    // Reconstucted birth point of neutral particle

   // ? Data from simulation.
   Int_t      label;        // Neutral mother label read from kinematics
   Int_t      pdg;          // PDG code of mother
   Int_t      d_label[2];   // Daughter labels ?? Rec labels present anyway.

   TEveRecV0() : status(), V_neg(), P_neg(), V_pos(), P_pos(),
                 V_ca(), V0_birth(), label(0), pdg(0)
   { d_label[0] = d_label[1] = 0; }
   virtual ~TEveRecV0() {}

   ClassDef(TEveRecV0, 1); // Reconstructed V0 (also used in VSD).
};

/******************************************************************************/
/******************************************************************************/

// Missing primary vertex.

// Missing TEveMCRecCrossRef, RecInfo.

class TEveMCRecCrossRef : public TObject
{
public:
   Bool_t       is_rec;   // is reconstructed
   Bool_t       has_V0;
   Bool_t       has_kink;
   Int_t        label;
   Int_t        n_hits;
   Int_t        n_clus;

   TEveMCRecCrossRef() : is_rec(false), has_V0(false), has_kink(false),
                         label(0), n_hits(0), n_clus(0) {}
   virtual ~TEveMCRecCrossRef() {}

   ClassDef(TEveMCRecCrossRef, 1); // Cross-reference of sim/rec data per particle (also used in VSD).
};

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
