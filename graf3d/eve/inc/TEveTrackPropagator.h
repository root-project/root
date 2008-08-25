// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTrackPropagator
#define ROOT_TEveTrackPropagator

#include "TEveVSDStructs.h"
#include "TEveUtil.h"
#include "TEveElement.h"
#include "TMarker.h"

#include <vector>


class TEveMagField
{
protected:
   Bool_t  fFieldConstant;

public:
   TEveMagField(): fFieldConstant(kFALSE){}
   virtual ~TEveMagField(){}

   virtual Bool_t IsConst() const {return fFieldConstant;};

   virtual void  PrintField(Float_t x, Float_t y, Float_t z) const
   {
      TEveVector b = GetField(x, y, z);
      printf("v(%f, %f, %f) B(%f, %f, %f) \n", x, y, z, b.fX, b.fY, b.fZ);
   }

   virtual TEveVector GetField(const TEveVector &v) const { return GetField(v.fX, v.fY, v.fZ);}

   virtual TEveVector GetField(Float_t x, Float_t y, Float_t z) const  =0;

   ClassDef(TEveMagField, 0); // Abstract interface to magnetic field
};


/**************************************************************************/
class TEveMagFieldConst: public TEveMagField
{
protected:
   TEveVector fB;

public:
   TEveMagFieldConst(Float_t x, Float_t y, Float_t z):TEveMagField() { fFieldConstant=kTRUE; fB.Set(x, y, z);}
   virtual ~TEveMagFieldConst(){}

   using   TEveMagField::GetField;
   virtual TEveVector GetField(Float_t /*x*/, Float_t /*y*/, Float_t /*z*/) const
   {
      return fB;
   }

   ClassDef(TEveMagFieldConst, 0); // Interface to constant magnetic field.
};

/**************************************************************************/
class TEveMagFieldDuo: public TEveMagField
{
protected:
   TEveVector fBIn;
   TEveVector fBOut;
   Float_t fR2;

public:
   TEveMagFieldDuo(Float_t r, Float_t bIn, Float_t bOut):TEveMagField(), fR2(r*r)
   {
      fFieldConstant=kFALSE;
      fBIn.Set(0, 0, bIn);
      fBOut.Set(0, 0, bOut);
   }
   virtual ~TEveMagFieldDuo(){}

   using   TEveMagField::GetField;

   virtual TEveVector GetField(Float_t x, Float_t y, Float_t /*z*/) const
   {
      return  ((x*x+y*y)<fR2) ? fBIn : fBOut;
   }

   virtual void  PrintField(Float_t x, Float_t y, Float_t z) const
   {
      TEveVector b = GetField(x, y, z);
   }

   ClassDef(TEveMagFieldDuo, 0); // Interface to magnetic field with two different values depending of radius.
};


/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/

class TEvePointSet;

class TEveTrackPropagator: public TEveElementList,
                           public TEveRefBackPtr
{
   friend class TEveTrackPropagatorSubEditor;

public:
   struct Helix_t
   {
      Int_t   fCharge;   // set in init track
      Float_t fMinAng;   // Minimal angular step between two helix points.
      Float_t fDelta;    // Maximal error at the mid-point of the line connecting to helix points.

      Float_t fPhi;      // accumulated angle to check fMaxOrbs by propagator
      Bool_t  fValid;    // corner case pT~0 or B~0, possible in variable mag field

      /**************************************************************************/

      // helix parameters
      Float_t fLam;         // momentum ratio pT/pZ
      Float_t fR;           // helix radius in cm
      Float_t fPhiStep;     // caluclated from fMinAng and fDelta
      Float_t fSin,  fCos;  // current sin, cos

      // cached 
      TEveVector fB;        // current magnetic field, cached 
      TEveVector fE1, fE2, fE3; // base vectors: E1 -> B dir, E2->pT dir, E3 = E1xE2
      TEveVector fPt, fPl;  // transverse, longitudinal momentum
      Float_t fPtMag;       // mag of p transverse
      Float_t fPlDir;       // momenum is in or oposite mag field
      Float_t fTStepSize;    // traverse step size in cm

      /**************************************************************************/


      Helix_t();
      ~Helix_t(){}

      void Update(const TEveVector& p, const TEveVector& b, Bool_t fullUpdate, Float_t fraction = -1);
      void Step(const TEveVector4& v, const TEveVector& p, TEveVector4& vOut, TEveVector& pOut);

      Float_t GetStepSize()  {return fTStepSize*TMath::Sqrt(1+fLam*fLam);}
      Float_t GetStepSize2() {return fTStepSize* fTStepSize*(1+fLam*fLam);}
   };

private:
   TEveTrackPropagator(const TEveTrackPropagator&);            // Not implemented
   TEveTrackPropagator& operator=(const TEveTrackPropagator&); // Not implemented

protected:
   //----------------------------------
   // Track extrapolation configuration
   TEveMagField*            fMagFieldObj;

   // TEveTrack limits
   Float_t                  fMaxR;          // Max radius for track extrapolation
   Float_t                  fMaxZ;          // Max z-coordinate for track extrapolation.
   Int_t                    fNMax;          // max steps
   // Helix limits
   Float_t                  fMaxOrbs;       // Maximal angular path of tracks' orbits (1 ~ 2Pi).

   // Path-mark / first-vertex control
   Bool_t                   fEditPathMarks; // Show widgets for path-mark control in GUI editor.
   Bool_t                   fFitDaughters;  // Pass through daughter creation points when extrapolating a track.
   Bool_t                   fFitReferences; // Pass through given track-references when extrapolating a track.
   Bool_t                   fFitDecay;      // Pass through decay point when extrapolating a track.
   Bool_t                   fFitCluster2Ds; // Pass through 2D-clusters when extrapolating a track.
   Bool_t                   fRnrDaughters;  // Render daughter path-marks.
   Bool_t                   fRnrReferences; // Render track-reference path-marks.
   Bool_t                   fRnrDecay;      // Render decay path-marks.
   Bool_t                   fRnrCluster2Ds; // Render 2D-clusters.
   Bool_t                   fRnrFV;         // Render first vertex.
   TMarker                  fPMAtt;         // Marker attributes for rendering of path-marks.
   TMarker                  fFVAtt;         // Marker attributes for fits vertex.

   //------------------------------------
   // propagation, state of current track

   std::vector<TEveVector4> fPoints;        // calculated point
   TEveVector               fV;             // start vertex
   Helix_t                  fH;             // helix

   void    RebuildTracks();
   void    StepHelix(TEveVector4 &v, TEveVector &p, TEveVector4 &vOut, TEveVector &pOut);

   Bool_t  HelixToVertex(TEveVector& v, TEveVector& p);
   void    HelixToBounds(TEveVector& p);

   Bool_t  LineToVertex (TEveVector& v);
   void    LineToBounds (TEveVector& p);

   Bool_t  HelixIntersectPlane(const TEveVector& p, const TEveVector& point, const TEveVector& normal,
                               TEveVector& itsect);
   Bool_t  LineIntersectPlane(const TEveVector& p, const TEveVector& point, const TEveVector& normal,
                              TEveVector& itsect);

   Bool_t PointOverVertex(const TEveVector4& v0, const TEveVector4& v);

public:
   TEveTrackPropagator(const Text_t* n="TEveTrackPropagator", const Text_t* t="",
                       TEveMagField* field=0);
   virtual ~TEveTrackPropagator();

   virtual void ElementChanged(Bool_t update_scenes=kTRUE, Bool_t redraw=kFALSE);

   // propagation
   void   InitTrack(TEveVector &v, TEveVector &p, Float_t beta, Int_t charge);
   void   ResetTrack();
   void   GoToBounds(TEveVector& p);
   Bool_t GoToVertex(TEveVector& v, TEveVector& p);

   Bool_t IntersectPlane(const TEveVector& p, const TEveVector& point, const TEveVector& normal,
                         TEveVector& itsect);

   void   FillPointSet(TEvePointSet* ps) const;

   void   SetMagField(Float_t bX, Float_t bY, Float_t bZ);
   void   SetMagField(Float_t b) { SetMagField(0.f, 0.f, b); }
   void   SetMagFieldObj(TEveMagField * x);

   void   SetMaxR(Float_t x);
   void   SetMaxZ(Float_t x);
   void   SetMaxOrbs(Float_t x);
   void   SetMinAng(Float_t x);
   void   SetDelta(Float_t x);

   void   SetEditPathMarks(Bool_t x) { fEditPathMarks = x; }
   void   SetRnrDaughters(Bool_t x);
   void   SetRnrReferences(Bool_t x);
   void   SetRnrDecay(Bool_t x);
   void   SetRnrCluster2Ds(Bool_t x);
   void   SetFitDaughters(Bool_t x);
   void   SetFitReferences(Bool_t x);
   void   SetFitDecay(Bool_t x);
   void   SetFitCluster2Ds(Bool_t x);
   void   SetRnrFV(Bool_t x) { fRnrFV = x; }

   TEveVector GetMagField(Float_t x, Float_t y, Float_t z) {return fMagFieldObj->GetField(x, y, z);}
   void PrintMagField(Float_t x, Float_t y, Float_t z) const;

   Float_t GetMaxR()     const { return fMaxR;     }
   Float_t GetMaxZ()     const { return fMaxZ;     }
   Float_t GetMaxOrbs()  const { return fMaxOrbs;  }
   Float_t GetMinAng()   const { return fH.fMinAng;   }
   Float_t GetDelta()    const { return fH.fDelta;    }

   Bool_t  GetEditPathMarks() const { return fEditPathMarks; }
   Bool_t  GetRnrDaughters()  const { return fRnrDaughters;  }
   Bool_t  GetRnrReferences() const { return fRnrReferences; }
   Bool_t  GetRnrDecay()      const { return fRnrDecay;      }
   Bool_t  GetRnrCluster2Ds() const { return fRnrCluster2Ds; }
   Bool_t  GetFitDaughters()  const { return fFitDaughters;  }
   Bool_t  GetFitReferences() const { return fFitReferences; }
   Bool_t  GetFitDecay()      const { return fFitDecay;      }
   Bool_t  GetFitCluster2Ds() const { return fFitCluster2Ds; }
   Bool_t  GetRnrFV()         const { return fRnrFV;         }

   TMarker& RefPMAtt() { return fPMAtt; }
   TMarker& RefFVAtt() { return fFVAtt; }

   static Bool_t IsOutsideBounds(const TEveVector& point, Float_t maxRsqr, Float_t maxZ);

   static Float_t             fgDefMagField; // Default value for constant solenoid magnetic field.
   static const Float_t       fgkB2C;        // Constant for conversion of momentum to curvature.
   static TEveTrackPropagator fgDefStyle;    // Default track render-style.

   ClassDef(TEveTrackPropagator, 0); // Calculates path of a particle taking into account special path-marks and imposed boundaries.
};


inline Bool_t TEveTrackPropagator::IsOutsideBounds(const TEveVector& point,
                                                   Float_t           maxRsqr,
                                                   Float_t           maxZ)
{
   // Return true if point% is outside of cylindrical bounds detrmined by
   // square radius and z.

   return TMath::Abs(point.fZ) > maxZ ||
          point.fX*point.fX + point.fY*point.fY > maxRsqr;
}

inline Bool_t TEveTrackPropagator::PointOverVertex(const TEveVector4 &v0,
                                                   const TEveVector4 &v)
{
   Float_t dotV = fH.fB.fX*(v0.fX-v.fX) 
                + fH.fB.fY*(v0.fY-v.fY) 
                + fH.fB.fZ*(v0.fZ-v.fZ);

   //printf("PointOverVertex pDotB %f  %f \n", fH.fPlDir , dotV);
   return (fH.fPlDir > 0 && dotV < 0) || (fH.fPlDir < 0 && dotV >0);
}

#endif
