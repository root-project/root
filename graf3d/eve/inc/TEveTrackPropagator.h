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

#include "TEveVector.h"
#include "TEvePathMark.h"
#include "TEveUtil.h"
#include "TEveElement.h"
#include "TMarker.h"

#include <vector>

class TEvePointSet;


//==============================================================================
// TEveMagField
//==============================================================================

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
   virtual TEveVector GetField(Float_t x, Float_t y, Float_t z) const = 0;
   virtual Float_t GetMaxFieldMag() const { return 4; } // not abstract because of backward compatibility

   ClassDef(TEveMagField, 0); // Abstract interface to magnetic field
};


//==============================================================================
// TEveMagFieldConst
//==============================================================================

class TEveMagFieldConst : public TEveMagField
{
protected:
   TEveVector fB;

public:
   TEveMagFieldConst(Float_t x, Float_t y, Float_t z) : TEveMagField(), fB(x, y, z)
   { fFieldConstant = kTRUE; }
   virtual ~TEveMagFieldConst() {}

   using   TEveMagField::GetField;
   virtual TEveVector GetField(Float_t /*x*/, Float_t /*y*/, Float_t /*z*/) const { return fB; }
   virtual Float_t GetMaxFieldMag() const { return fB.Mag(); };

   ClassDef(TEveMagFieldConst, 0); // Interface to constant magnetic field.
};


//==============================================================================
// TEveMagFieldDuo
//==============================================================================

class TEveMagFieldDuo : public TEveMagField
{
protected:
   TEveVector fBIn;
   TEveVector fBOut;
   Float_t    fR2;

public:
   TEveMagFieldDuo(Float_t r, Float_t bIn, Float_t bOut) : TEveMagField(),
     fBIn(0,0,bIn), fBOut(0,0,bOut), fR2(r*r)
   {
      fFieldConstant = kFALSE;
   }
   virtual ~TEveMagFieldDuo() {}

   using   TEveMagField::GetField;
   virtual TEveVector GetField(Float_t x, Float_t y, Float_t /*z*/) const
   { return  ((x*x+y*y)<fR2) ? fBIn : fBOut; }
   virtual Float_t GetMaxFieldMag() const
   { Float_t b1 = fBIn.Mag(), b2 = fBOut.Mag(); return b1 > b2 ? b1 : b2; }

   ClassDef(TEveMagFieldDuo, 0); // Interface to magnetic field with two different values depending of radius.
};


//==============================================================================
// TEveTrackPropagator
//==============================================================================

class TEveTrackPropagator : public TEveElementList,
                            public TEveRefBackPtr
{
   friend class TEveTrackPropagatorSubEditor;

public:
   struct Helix_t
   {
      Int_t   fCharge;   // Charge of tracked particle.
      Float_t fMaxAng;   // Maximum step angle.
      Float_t fMaxStep;  // Maximum allowed step size.
      Float_t fDelta;    // Maximum error in the middle of the step.

      Float_t fPhi;      // Accumulated angle to check fMaxOrbs by propagator.
      Bool_t  fValid;    // Corner case pT~0 or B~0, possible in variable mag field.

      // ----------------------------------------------------------------

      // helix parameters
      Float_t fLam;         // Momentum ratio pT/pZ.
      Float_t fR;           // Helix radius in cm.
      Float_t fPhiStep;     // Caluclated from fMinAng and fDelta.
      Float_t fSin, fCos;   // Current sin/cos(phistep).

      // Runge-Kutta parameters
      Float_t fRKStep;      // Step for Runge-Kutta.

      // cached
      TEveVector fB;        // Current magnetic field, cached.
      TEveVector fE1, fE2, fE3; // Base vectors: E1 -> B dir, E2->pT dir, E3 = E1xE2.
      TEveVector fPt, fPl;  // Transverse and longitudinal momentum.
      Float_t fPtMag;       // Magnitude of pT.
      Float_t fPlMag;       // Momentum parallel to mag field.
      Float_t fLStep;       // Transverse step arc-length in cm.

      // ----------------------------------------------------------------

      Helix_t();

      void UpdateCommon(const TEveVector & p, const TEveVector& b);
      void UpdateHelix(const TEveVector & p, const TEveVector& b, Bool_t full_update, Bool_t enforce_max_step);
      void UpdateRK   (const TEveVector & p, const TEveVector& b);

      void Step(const TEveVector4& v, const TEveVector& p, TEveVector4& vOut, TEveVector& pOut);

      Float_t GetStep()  { return fLStep * TMath::Sqrt(1 + fLam*fLam); }
      Float_t GetStep2() { return fLStep * fLStep * (1 + fLam*fLam);   }
   };

   enum EStepper_e    { kHelix, kRungeKutta };

   enum EProjTrackBreaking_e { kPTB_Break, kPTB_UseFirstPointPos, kPTB_UseLastPointPos };

private:
   TEveTrackPropagator(const TEveTrackPropagator&);            // Not implemented
   TEveTrackPropagator& operator=(const TEveTrackPropagator&); // Not implemented

protected:
   EStepper_e               fStepper;

   TEveMagField*            fMagFieldObj;
   Bool_t                   fOwnMagFiledObj;

   // Track extrapolation limits
   Float_t                  fMaxR;          // Max radius for track extrapolation
   Float_t                  fMaxZ;          // Max z-coordinate for track extrapolation.
   Int_t                    fNMax;          // Max steps
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

   // Handling of discontinuities in projections
   UChar_t                  fProjTrackBreaking; // Handling of projected-track breaking.
   Bool_t                   fRnrPTBMarkers;     // Render break-points on tracks.
   TMarker                  fPTBAtt;            // Marker attributes for track break-points.

   // ----------------------------------------------------------------

   // Propagation, state of current track
   std::vector<TEveVector4> fPoints;        // Calculated point.
   TEveVector               fV;             // Start vertex.
   Helix_t                  fH;             // Helix.

   void    RebuildTracks();
   void    Update(const TEveVector4& v, const TEveVector& p, Bool_t full_update=kFALSE, Bool_t enforce_max_step=kFALSE);
   void    Step(const TEveVector4 &v, const TEveVector &p, TEveVector4 &vOut, TEveVector &pOut);

   Bool_t  LoopToVertex(TEveVector& v, TEveVector& p);
   void    LoopToBounds(TEveVector& p);

   Bool_t  LineToVertex (TEveVector& v);
   void    LineToBounds (TEveVector& p);

   void    StepRungeKutta(Double_t step, Double_t* vect, Double_t* vout);

   Bool_t  HelixIntersectPlane(const TEveVector& p, const TEveVector& point, const TEveVector& normal,
                               TEveVector& itsect);
   Bool_t  LineIntersectPlane(const TEveVector& p, const TEveVector& point, const TEveVector& normal,
                              TEveVector& itsect);

   Bool_t PointOverVertex(const TEveVector4& v0, const TEveVector4& v, Float_t* p=0);

public:
   TEveTrackPropagator(const char* n="TEveTrackPropagator", const char* t="",
                       TEveMagField* field=0, Bool_t own_field=kTRUE);
   virtual ~TEveTrackPropagator();

   virtual void OnZeroRefCount();

   virtual void CheckReferenceCount(const TEveException& eh="TEveElement::CheckReferenceCount ");

   virtual void ElementChanged(Bool_t update_scenes=kTRUE, Bool_t redraw=kFALSE);

   // propagation
   void   InitTrack(TEveVector& v, Int_t charge);
   void   ResetTrack();
   void   GoToBounds(TEveVector& p);
   Bool_t GoToVertex(TEveVector& v, TEveVector& p);

   Bool_t IntersectPlane(const TEveVector& p, const TEveVector& point, const TEveVector& normal,
                         TEveVector& itsect);

   void   FillPointSet(TEvePointSet* ps) const;

   void   SetStepper(EStepper_e s) { fStepper = s; }

   void   SetMagField(Float_t bX, Float_t bY, Float_t bZ);
   void   SetMagField(Float_t b) { SetMagField(0.f, 0.f, b); }
   void   SetMagFieldObj(TEveMagField* field, Bool_t own_field=kTRUE);

   void   SetMaxR(Float_t x);
   void   SetMaxZ(Float_t x);
   void   SetMaxOrbs(Float_t x);
   void   SetMinAng(Float_t x);
   void   SetMaxAng(Float_t x);
   void   SetMaxStep(Float_t x);
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
   void   SetRnrFV(Bool_t x);
   void   SetProjTrackBreaking(UChar_t x);
   void   SetRnrPTBMarkers(Bool_t x);

   TEveVector GetMagField(Float_t x, Float_t y, Float_t z) { return fMagFieldObj->GetField(x, y, z); }
   void PrintMagField(Float_t x, Float_t y, Float_t z) const;

   EStepper_e   GetStepper()  const { return fStepper;}

   Float_t GetMaxR()     const { return fMaxR;     }
   Float_t GetMaxZ()     const { return fMaxZ;     }
   Float_t GetMaxOrbs()  const { return fMaxOrbs;  }
   Float_t GetMinAng()   const;
   Float_t GetMaxAng()   const { return fH.fMaxAng;   }
   Float_t GetMaxStep()  const { return fH.fMaxStep;  }
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
   UChar_t GetProjTrackBreaking() const { return fProjTrackBreaking; }
   Bool_t  GetRnrPTBMarkers()     const { return fRnrPTBMarkers; }

   TMarker& RefPMAtt()  { return fPMAtt; }
   TMarker& RefFVAtt()  { return fFVAtt; }
   TMarker& RefPTBAtt() { return fPTBAtt; }
   

   static Bool_t IsOutsideBounds(const TEveVector& point, Float_t maxRsqr, Float_t maxZ);

   static Float_t             fgDefMagField; // Default value for constant solenoid magnetic field.
   static const Float_t       fgkB2C;        // Constant for conversion of momentum to curvature.
   static TEveTrackPropagator fgDefault;     // Default track propagator.

   static Float_t             fgEditorMaxR;  // Max R that can be set in GUI editor.
   static Float_t             fgEditorMaxZ;  // Max Z that can be set in GUI editor.

   ClassDef(TEveTrackPropagator, 0); // Calculates path of a particle taking into account special path-marks and imposed boundaries.
};

//______________________________________________________________________________
inline Bool_t TEveTrackPropagator::IsOutsideBounds(const TEveVector& point,
                                                   Float_t           maxRsqr,
                                                   Float_t           maxZ)
{
   // Return true if point% is outside of cylindrical bounds detrmined by
   // square radius and z.

   return TMath::Abs(point.fZ) > maxZ ||
          point.fX*point.fX + point.fY*point.fY > maxRsqr;
}

//______________________________________________________________________________
inline Bool_t TEveTrackPropagator::PointOverVertex(const TEveVector4 &v0,
                                                   const TEveVector4 &v,
                                                   Float_t           *p)
{
   static const Float_t kMinPl = 1e-5;

   TEveVector dv; dv.Sub(v0, v);

   Float_t dotV;

   if (TMath::Abs(fH.fPlMag) > kMinPl)
   {
      // Use longitudinal momentum to determine crossing point.
      // Works ok for spiraling helices, also for loopers.

      dotV = fH.fE1.Dot(dv);
      if (fH.fPlMag < 0)
         dotV = -dotV;
   }
   else
   {
      // Use full momentum, which is pT, under this conditions.

      dotV = fH.fE2.Dot(dv);
   }

   if (p)
      *p = dotV;

   return dotV < 0;
}

#endif
