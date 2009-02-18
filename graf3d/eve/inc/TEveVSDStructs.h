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
#include "TMath.h"

#include "TParticle.h"

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
// TEveVector
/******************************************************************************/

class TEveVector
{
public:
   Float_t fX, fY, fZ; // Components of the vector.

   TEveVector() : fX(0), fY(0), fZ(0) {}
   TEveVector(const Float_t* v)  : fX(v[0]), fY(v[1]), fZ(v[2]) {}
   TEveVector(const Double_t* v) : fX(v[0]), fY(v[1]), fZ(v[2]) {}
   TEveVector(Float_t x, Float_t y, Float_t z) : fX(x), fY(y), fZ(z) {}
   virtual ~TEveVector() {}

   void Dump() const;

   operator const Float_t*() const { return &fX; }
   operator       Float_t*()       { return &fX; }

   TEveVector& operator *=(Float_t s)           { fX *= s;    fY *= s;    fZ *= s;    return *this; }
   TEveVector& operator +=(const TEveVector& v) { fX += v.fX; fY += v.fY; fZ += v.fZ; return *this; }
   TEveVector& operator -=(const TEveVector& v) { fX -= v.fX; fY -= v.fY; fZ -= v.fZ; return *this; }

   TEveVector operator + (const TEveVector &) const;
   TEveVector operator - (const TEveVector &) const;
   TEveVector operator * (Float_t a) const;

   Float_t& operator [] (Int_t indx);
   Float_t  operator [] (Int_t indx) const;

   const Float_t* Arr() const { return &fX; }
         Float_t* Arr()       { return &fX; }

   void Set(const Float_t*  v) { fX = v[0]; fY = v[1]; fZ = v[2]; }
   void Set(const Double_t* v) { fX = v[0]; fY = v[1]; fZ = v[2]; }
   void Set(Float_t  x, Float_t  y, Float_t  z) { fX = x; fY = y; fZ = z; }
   void Set(Double_t x, Double_t y, Double_t z) { fX = x; fY = y; fZ = z; }
   void Set(const TVector3& v)   { fX = v.x(); fY = v.y(); fZ = v.z(); }
   void Set(const TEveVector& v) { fX = v.fX;  fY = v.fY;  fZ = v.fZ;  }

   void NegateXYZ() { fX = - fX; fY = -fY; fZ = -fZ; }
   void Normalize(Float_t length=1);

   Float_t Phi()      const;
   Float_t Theta()    const;
   Float_t CosTheta() const;
   Float_t Eta()      const;

   Float_t Mag()  const { return TMath::Sqrt(fX*fX + fY*fY + fZ*fZ);}
   Float_t Mag2() const { return fX*fX + fY*fY + fZ*fZ;}

   Float_t Perp()  const { return TMath::Sqrt(fX*fX + fY*fY);}
   Float_t Perp2() const { return fX*fX + fY*fY;}
   Float_t R()     const { return Perp(); }

   Float_t Distance(const TEveVector& v) const;
   Float_t SquareDistance(const TEveVector& v) const;
   Float_t Dot(const TEveVector&a) const;

   TEveVector& Mult(const TEveVector& a, Float_t af)
   { fX = a.fX*af; fY = a.fY*af; fZ = a.fZ*af; return *this; }

   TEveVector Orthogonal() const;
   void OrthoNormBase(TEveVector& a, TEveVector& b) const;

   ClassDef(TEveVector, 1); // Float three-vector; a minimal Float_t copy of TVector3 used to represent points and momenta (also used in VSD).
};

//______________________________________________________________________________
inline Float_t TEveVector::Phi() const
{ return fX == 0.0 && fY == 0.0 ? 0.0 : TMath::ATan2(fY, fX); }

inline Float_t TEveVector::Theta() const
{ return fX == 0.0 && fY == 0.0 && fZ == 0.0 ? 0.0 : TMath::ATan2(Perp(), fZ); }

inline Float_t TEveVector::CosTheta() const
{ Float_t ptot = Mag(); return ptot == 0.0 ? 1.0 : fZ/ptot; }

inline Float_t TEveVector::Distance( const TEveVector& b) const
{
   return TMath::Sqrt((fX - b.fX)*(fX - b.fX) +
                      (fY - b.fY)*(fY - b.fY) +
                      (fZ - b.fZ)*(fZ - b.fZ));
}
inline Float_t TEveVector::SquareDistance(const TEveVector& b) const
{
   return ((fX - b.fX) * (fX - b.fX) +
           (fY - b.fY) * (fY - b.fY) +
           (fZ - b.fZ) * (fZ - b.fZ));
}

//______________________________________________________________________________
inline Float_t TEveVector::Dot(const TEveVector& a) const
{
   return a.fX*fX + a.fY*fY + a.fZ*fZ;
}

inline Float_t& TEveVector::operator [] (Int_t idx)
{ return (&fX)[idx]; }

inline Float_t TEveVector::operator [] (Int_t idx) const
{ return (&fX)[idx]; }


/******************************************************************************/
// TEveVector4
/******************************************************************************/

class TEveVector4 : public TEveVector
{
public:
   Float_t fT;

   TEveVector4()                    : TEveVector(),  fT(0) {}
   TEveVector4(const TEveVector& v) : TEveVector(v), fT(0) {}
   TEveVector4(Float_t x, Float_t y, Float_t z, Float_t t=0) :
      TEveVector(x, y, z), fT(t) {}
   virtual ~TEveVector4() {}

   void Dump() const;

   TEveVector4 operator + (const TEveVector4 & b)
   { return TEveVector4(fX + b.fX, fY + b.fY, fZ + b.fZ, fT + b.fT); }

   TEveVector4 operator - (const TEveVector4 & b)
   { return TEveVector4(fX - b.fX, fY - b.fY, fZ - b.fZ, fT - b.fT); }

   TEveVector4 operator * (Float_t a)
   { return TEveVector4(a*fX, a*fY, a*fZ, a*fT); }

   TEveVector4& operator += (const TEveVector4 & b)
   { fX += b.fX; fY += b.fY; fZ += b.fZ; fT += b.fT; return *this; }

   ClassDef(TEveVector4, 1); // Float four-vector.
};



/******************************************************************************/
// TEvePathMark
/******************************************************************************/

class TEvePathMark
{
public:
   enum EType_e   { kReference, kDaughter, kDecay, kCluster2D };

   EType_e     fType; // Mark-type.
   TEveVector  fV;    // Vertex.
   TEveVector  fP;    // Momentum.
   TEveVector  fE;    // Extra, meaning depends on fType.
   Float_t     fTime; // Time.

   TEvePathMark(EType_e type=kReference) :
      fType(type), fV(), fP(), fE(), fTime(0) {}

   TEvePathMark(EType_e type, const TEveVector& v, Float_t time=0) :
      fType(type), fV(v), fP(), fE(), fTime(time) {}

   TEvePathMark(EType_e type, const TEveVector& v, const TEveVector& p, Float_t time=0) :
      fType(type), fV(v), fP(p), fE(), fTime(time) {}

   TEvePathMark(EType_e type, const TEveVector& v, const TEveVector& p, const TEveVector& e, Float_t time=0) :
      fType(type), fV(v), fP(p), fE(e), fTime(time) {}

   virtual ~TEvePathMark() {}

   const char* TypeName();

   ClassDef(TEvePathMark, 1); // Special-point on track: position/momentum reference, daughter creation or decay (also used in VSD).
};

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

class TEveRecTrack : public TObject
{
public:
   Int_t       fLabel;       // Label of the track.
   Int_t       fIndex;       // Index of the track (in some source array).
   Int_t       fStatus;      // Status as exported from reconstruction.
   Int_t       fSign;        // Charge of the track.
   TEveVector  fV;           // Start vertex from reconstruction.
   TEveVector  fP;           // Reconstructed momentum at start vertex.
   Float_t     fBeta;        // Relativistic beta factor.

   // PID data missing

   TEveRecTrack() : fLabel(-1), fIndex(-1), fStatus(0), fSign(0), fV(), fP(), fBeta(0) {}
   virtual ~TEveRecTrack() {}

   Float_t Pt() { return fP.Perp(); }

   ClassDef(TEveRecTrack, 1); // Reconstructed track (also used in VSD).
};


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
