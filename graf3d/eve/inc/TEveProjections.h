// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveProjections
#define ROOT_TEveProjections

#include "TEveVector.h"
#include "TString.h"

#include <vector>

class TEveTrans;

//==============================================================================
// TEveProjection
//==============================================================================

class TEveProjection
{
public:
   enum EPType_e   { kPT_Unknown, kPT_RPhi, kPT_RhoZ, kPT_3D, kPT_End }; // projection type
   enum EPProc_e   { kPP_Plane, kPP_Distort, kPP_Full };                 // projection procedure
   enum EGeoMode_e { kGM_Unknown, kGM_Polygons, kGM_Segments };          // strategy for geometry projections

   struct PreScaleEntry_t
   {
      Float_t fMin, fMax;
      Float_t fOffset;
      Float_t fScale;

      PreScaleEntry_t() :
         fMin(0), fMax(0), fOffset(0), fScale(1) {}
      PreScaleEntry_t(Float_t min, Float_t max, Float_t off, Float_t scale) :
         fMin(min), fMax(max), fOffset(off), fScale(scale) {}

      virtual ~PreScaleEntry_t() {}

      ClassDef(PreScaleEntry_t, 0);
   };

   typedef std::vector<PreScaleEntry_t>           vPreScale_t;
   typedef std::vector<PreScaleEntry_t>::iterator vPreScale_i;

protected:
   EPType_e            fType;          // type
   EGeoMode_e          fGeoMode;       // strategy of polygon projection (what to try first)
   TString             fName;          // name

   TEveVector          fCenter;        // center of distortionprivate:

   bool                fDisplaceOrigin; // displace point before projection

   Bool_t              fUsePreScale;   // use pre-scaling
   vPreScale_t         fPreScales[3];  // scaling before the distortion

   Float_t             fDistortion;    // distortion
   Float_t             fFixR;          // radius from which scaling remains constant
   Float_t             fFixZ;          // z-coordinate from which scaling remains constant
   Float_t             fPastFixRFac;   // relative scaling factor beyond fFixR as 10^x
   Float_t             fPastFixZFac;   // relative scaling factor beyond fFixZ as 10^x
   Float_t             fScaleR;        // scale factor to keep projected radius at fFixR fixed
   Float_t             fScaleZ;        // scale factor to keep projected z-coordinate at fFixZ fixed
   Float_t             fPastFixRScale; // relative scaling beyond fFixR
   Float_t             fPastFixZScale; // relative scaling beyond fFixZ
   Float_t             fMaxTrackStep;  // maximum distance between two points on a track

   void PreScaleVariable(Int_t dim, Float_t& v);

public:
   TEveProjection();
   virtual ~TEveProjection() {}

   virtual Bool_t      Is2D() const = 0;
   virtual Bool_t      Is3D() const = 0;

   virtual void        ProjectPoint(Float_t& x, Float_t& y, Float_t& z, Float_t d, EPProc_e p = kPP_Full) = 0;

   void                ProjectPointfv(Float_t* v, Float_t d);
   void                ProjectPointdv(Double_t* v, Float_t d);
   void                ProjectVector(TEveVector& v, Float_t d);

   void                ProjectPointfv(const TEveTrans* t, const Float_t*  p, Float_t* v, Float_t d);
   void                ProjectPointdv(const TEveTrans* t, const Double_t* p, Double_t* v, Float_t d);
   void                ProjectVector(const TEveTrans* t, TEveVector& v, Float_t d);

   const   Char_t*     GetName() const            { return fName.Data(); }
   void                SetName(const Char_t* txt) { fName = txt; }

   virtual void        SetCenter(TEveVector& v) { fCenter = v; }
   virtual Float_t*    GetProjectedCenter();
  
   void                SetDisplaceOrigin(bool);
   Bool_t              GetDisplaceOrigin() const { return fDisplaceOrigin; }

   void                SetType(EPType_e t)        { fType = t; }
   EPType_e            GetType() const            { return fType; }

   void                SetGeoMode(EGeoMode_e m)   { fGeoMode = m; }
   EGeoMode_e          GetGeoMode() const         { return fGeoMode; }

   Bool_t   GetUsePreScale() const   { return fUsePreScale; }
   void     SetUsePreScale(Bool_t x) { fUsePreScale = x; }

   void     PreScalePoint(Float_t& x, Float_t& y);
   void     PreScalePoint(Float_t& x, Float_t& y, Float_t& z);
   void     AddPreScaleEntry(Int_t coord, Float_t max_val, Float_t scale);
   void     ChangePreScaleEntry(Int_t coord, Int_t entry, Float_t new_scale);
   void     ClearPreScales();

   void     SetDistortion(Float_t d);
   Float_t  GetDistortion() const { return fDistortion; }
   Float_t  GetFixR() const { return fFixR; }
   Float_t  GetFixZ() const { return fFixZ; }
   void     SetFixR(Float_t x);
   void     SetFixZ(Float_t x);
   Float_t  GetPastFixRFac() const { return fPastFixRFac; }
   Float_t  GetPastFixZFac() const { return fPastFixZFac; }
   void     SetPastFixRFac(Float_t x);
   void     SetPastFixZFac(Float_t x);
   Float_t  GetMaxTrackStep() const    { return fMaxTrackStep; }
   void     SetMaxTrackStep(Float_t x) { fMaxTrackStep = TMath::Max(x, 1.0f); }

   virtual Bool_t      HasSeveralSubSpaces() const { return kFALSE; }
   virtual Bool_t      AcceptSegment(TEveVector&, TEveVector&, Float_t /*tolerance*/) const { return kTRUE; }
   virtual Int_t       SubSpaceId(const TEveVector&) const { return 0; }
   virtual Bool_t      IsOnSubSpaceBoundrary(const TEveVector&) const { return kFALSE; }
   virtual void        BisectBreakPoint(TEveVector& vL, TEveVector& vR, Float_t eps_sqr=1e-10f);
   virtual void        SetDirectionalVector(Int_t screenAxis, TEveVector& vec);

   // utils to draw axis
   TEveVector          GetOrthogonalCenter(int idx, TEveVector& out);
   virtual Float_t     GetValForScreenPos(Int_t ax, Float_t value);
   virtual Float_t     GetScreenVal(Int_t ax, Float_t value);
   Float_t             GetScreenVal(Int_t i, Float_t x, TEveVector& dirVec, TEveVector& oCenter);
   Float_t             GetLimit(Int_t i, Bool_t pos);


   static   Float_t    fgEps;    // resolution of projected points
   static   Float_t    fgEpsSqr; // square of resolution of projected points

   ClassDef(TEveProjection, 0); // Base for specific classes that implement non-linear projections.
};


//==============================================================================
// TEveRhoZProjection
//==============================================================================

class TEveRhoZProjection: public TEveProjection
{
private:
   TEveVector   fProjectedCenter; // projected center of distortion.

public:
   TEveRhoZProjection();
   virtual ~TEveRhoZProjection() {}

   virtual Bool_t      Is2D() const { return kTRUE;  }
   virtual Bool_t      Is3D() const { return kFALSE; }

   virtual void        ProjectPoint(Float_t& x, Float_t& y, Float_t& z, Float_t d, EPProc_e proc = kPP_Full);

   virtual void        SetCenter(TEveVector& v); 
   virtual Float_t*    GetProjectedCenter() { return fProjectedCenter.Arr(); }

   virtual Bool_t      HasSeveralSubSpaces() const { return kTRUE; }
   virtual Bool_t      AcceptSegment(TEveVector& v1, TEveVector& v2, Float_t tolerance) const;
   virtual Int_t       SubSpaceId(const TEveVector& v) const;
   virtual Bool_t      IsOnSubSpaceBoundrary(const TEveVector& v) const;
   virtual void        SetDirectionalVector(Int_t screenAxis, TEveVector& vec);

   ClassDef(TEveRhoZProjection, 0); // Rho/Z non-linear projection.
};


//==============================================================================
// TEveRPhiProjection
//==============================================================================

class TEveRPhiProjection : public TEveProjection
{
public:
   TEveRPhiProjection();
   virtual ~TEveRPhiProjection() {}

   virtual Bool_t Is2D() const { return kTRUE;  }
   virtual Bool_t Is3D() const { return kFALSE; }

   virtual void   ProjectPoint(Float_t& x, Float_t& y, Float_t& z, Float_t d, EPProc_e proc = kPP_Full);

   ClassDef(TEveRPhiProjection, 0); // XY non-linear projection.
};


//==============================================================================
// TEve3DProjection
//==============================================================================

class TEve3DProjection : public TEveProjection
{
public:
   TEve3DProjection();
   virtual ~TEve3DProjection() {}

   virtual Bool_t Is2D() const { return kFALSE; }
   virtual Bool_t Is3D() const { return kTRUE;  }

   virtual void   ProjectPoint(Float_t& x, Float_t& y, Float_t& z, Float_t d, EPProc_e proc = kPP_Full);

   ClassDef(TEve3DProjection, 0); // 3D scaling "projection"
};

// AMT: temporary workaround till root pactches are integrated in CMSSW 	 
#define TEVEPROJECTIONS_DISPLACE_ORIGIN_MODE	 
	 
#endif
