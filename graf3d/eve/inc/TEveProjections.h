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

#include "TEveVSDStructs.h"

#include <vector>

////////////////////////////////////////////////////////////////
//                                                            //
// TEveProjection                                             //
//                                                            //
////////////////////////////////////////////////////////////////

class TEveProjection
{
public:
   enum EPType_e   { kPT_Unknown, kPT_RPhi, kPT_RhoZ, kPT_End };// type
   enum EPProc_e   { kPP_Plane, kPP_Distort, kPP_Full };        // procedure
   enum EGeoMode_e { kGM_Unknown, kGM_Polygons, kGM_Segments }; // reconstruction of geometry

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

   TEveVector          fCenter;        // center of distortion
   TEveVector          fZeroPosVal;    // projected origin (0, 0, 0)

   Bool_t              fUsePreScale;   // use pre-scaling
   vPreScale_t         fPreScales[2];  // scaling before the distortion

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

   TEveVector          fLowLimit;      // convergence of point +infinity
   TEveVector          fUpLimit;       // convergence of point -infinity

public:
   TEveProjection();
   virtual ~TEveProjection() {}

   virtual   void      ProjectPoint(Float_t&, Float_t&, Float_t&, EPProc_e p = kPP_Full ) = 0;
   virtual   void      ProjectPointFv(Float_t* v) { ProjectPoint(v[0], v[1], v[2]); }
   virtual   void      ProjectVector(TEveVector& v);

   const     char*     GetName() { return fName.Data(); }
   void                SetName(const char* txt) { fName = txt; }

   virtual void        SetCenter(TEveVector& v) { fCenter = v; UpdateLimit(); }
   virtual Float_t*    GetProjectedCenter() { return fCenter.Arr(); }

   void                SetType(EPType_e t) { fType = t; }
   EPType_e            GetType() { return fType; }

   void                SetGeoMode(EGeoMode_e m) { fGeoMode = m; }
   EGeoMode_e          GetGeoMode() { return fGeoMode; }

   virtual void        UpdateLimit();

   Bool_t   GetUsePreScale() const   { return fUsePreScale; }
   void     SetUsePreScale(Bool_t x) { fUsePreScale = x; }

   void     PreScalePoint(Float_t& x, Float_t& y);
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

   virtual   Bool_t    AcceptSegment(TEveVector&, TEveVector&, Float_t /*tolerance*/) { return kTRUE; }
   virtual   void      SetDirectionalVector(Int_t screenAxis, TEveVector& vec);

   // utils to draw axis
   virtual Float_t     GetValForScreenPos(Int_t ax, Float_t value);
   virtual Float_t     GetScreenVal(Int_t ax, Float_t value);
   Float_t             GetLimit(Int_t i, Bool_t pos) { return pos ? fUpLimit[i] : fLowLimit[i]; }

   static   Float_t    fgEps;  // resolution of projected points

   ClassDef(TEveProjection, 0); // Base for specific classes that implement non-linear projections.
};


////////////////////////////////////////////////////////////////
//                                                            //
// TEveRhoZProjection                                         //
//                                                            //
////////////////////////////////////////////////////////////////

class TEveRhoZProjection: public TEveProjection
{
private:
   TEveVector   fProjectedCenter; // projected center of distortion.

public:
   TEveRhoZProjection();
   virtual ~TEveRhoZProjection() {}

   virtual   void      ProjectPoint(Float_t& x, Float_t& y, Float_t& z, EPProc_e proc = kPP_Full);

   virtual   void      SetCenter(TEveVector& center);
   virtual   Float_t*  GetProjectedCenter() { return fProjectedCenter.Arr(); }

   virtual   void      UpdateLimit();

   virtual   Bool_t    AcceptSegment(TEveVector& v1, TEveVector& v2, Float_t tolerance);
   virtual   void      SetDirectionalVector(Int_t screenAxis, TEveVector& vec);

   ClassDef(TEveRhoZProjection, 0); // Rho/Z non-linear projection.
};


////////////////////////////////////////////////////////////////
//                                                            //
// TEveRPhiProjection                                         //
//                                                            //
////////////////////////////////////////////////////////////////

class TEveRPhiProjection : public TEveProjection
{
public:
   TEveRPhiProjection();
   virtual ~TEveRPhiProjection() {}

   virtual void ProjectPoint(Float_t& x, Float_t& y, Float_t& z, EPProc_e proc = kPP_Full);

   ClassDef(TEveRPhiProjection, 0); // XY non-linear projection.
};

#endif
