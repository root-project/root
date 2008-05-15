// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveCalo
#define ROOT_TEveCalo

#include "TEveElement.h"
#include "TEveProjectionBases.h"
#include "TEveProjectionManager.h"

#include "TAtt3D.h"
#include "TAttBBox.h"
#include "TEveCaloData.h"

class TClass;
class TEveRGBAPalette;

class TEveCaloViz : public TEveElement,
                    public TNamed,
                    public TAtt3D,
                    public TAttBBox,
                    public TEveProjectable
{
   friend class TEveCaloVizEditor;

private:
   TEveCaloViz(const TEveCaloViz&);        // Not implemented
   TEveCaloViz& operator=(const TEveCaloViz&); // Not implemented

protected:
   TEveCaloData* fData;  // event data reference

   Float_t      fEtaLowLimit;
   Float_t      fEtaHighLimit;
   Float_t      fEtaMin;
   Float_t      fEtaMax;

   Float_t      fPhi;
   Float_t      fPhiRng;

   Float_t      fBarrelRadius;  // barrel raidus in cm
   Float_t      fEndCapPos;     // end cap z coordinate in cm

   Float_t      fCellZScale;

   Bool_t            fValueIsColor;   // Interpret signal value as RGBA color.
   TEveRGBAPalette*  fPalette;        // Pointer to signal-color palette.

   Bool_t            fCacheOK;        // is list of list of cell ids valid

   void AssignCaloVizParameters(TEveCaloViz* cv);

   void SetupColorHeight(Float_t value, Int_t slice, Float_t& height, Bool_t &viz) const;

public:
   TEveCaloViz(const Text_t* n="TEveCaloViz", const Text_t* t="");
   TEveCaloViz(TEveCaloData* data, const Text_t* n="TEveCaloViz", const Text_t* t="");

   virtual ~TEveCaloViz();

   void InvalidateCache() { fCacheOK = kFALSE; ResetBBox(); }

   TEveCaloData* GetData() const { return fData; }
   virtual void  SetData(TEveCaloData* d);

   Float_t GetBarrelRadius() const { return fBarrelRadius; }
   void SetBarrelRadius(Float_t r) { fBarrelRadius = r; ResetBBox(); }
   Float_t GetEndCapPos   () const { return fEndCapPos; }
   void SetEndCapPos   (Float_t z) { fEndCapPos = z; ResetBBox(); }

   virtual void    SetCellZScale(Float_t s) { fCellZScale = s; ResetBBox(); }
   virtual Float_t GetDefaultCellHeight() const { return fBarrelRadius*fCellZScale; }

   Float_t GetTransitionEta() const;
   Float_t GetTransitionTheta() const;

   TEveRGBAPalette* GetPalette() const { return fPalette; }
   void             SetPalette(TEveRGBAPalette* p);
   TEveRGBAPalette* AssertPalette();


   void SetEta(Float_t l, Float_t u) { fEtaMin=l; fEtaMax=u; InvalidateCache(); }
   void SetEtaLimits(Float_t l, Float_t h) { fEtaLowLimit=l; fEtaHighLimit =h; InvalidateCache(); }

   void SetPhi(Float_t x)    { fPhi    = x; InvalidateCache(); }
   void SetPhiRng(Float_t r) { fPhiRng = r; InvalidateCache(); }
   void SetPhiWithRng(Float_t x, Float_t r) { fPhi = x; fPhiRng = r; InvalidateCache(); }


   virtual void ResetCache() = 0;

   virtual void Paint(Option_t* option="");

   virtual TClass* ProjectedClass() const;

   ClassDef(TEveCaloViz, 0); // Base-class for visualization of calorimeter eventdata.
};

/**************************************************************************/
/**************************************************************************/

class TEveCalo3D : public TEveCaloViz
{
   friend class TEveCalo3DGL;
private:
   TEveCalo3D(const TEveCalo3D&);            // Not implemented
   TEveCalo3D& operator=(const TEveCalo3D&); // Not implemented

protected:
   TEveCaloData::vCellId_t fCellList;

public:
   TEveCalo3D(const Text_t* n="TEveCalo3D", const Text_t* t=""):TEveCaloViz(n, t){ fCellZScale = 0.2;}
   TEveCalo3D(TEveCaloData* data): TEveCaloViz(data) { SetElementName("TEveCalo3D"); fCellZScale = 0.2;}
   virtual ~TEveCalo3D() {}
   virtual void ComputeBBox();

   virtual void ResetCache();

   ClassDef(TEveCalo3D, 0); // Class for 3D visualization of calorimeter event data.
};

/**************************************************************************/
/**************************************************************************/

class TEveCalo2D : public TEveCaloViz,
                   public TEveProjected
{
   friend class TEveCalo2DGL;
private:
   TEveCalo2D(const TEveCalo2D&);            // Not implemented
   TEveCalo2D& operator=(const TEveCalo2D&); // Not implemented

   TEveProjection::EPType_e  fOldProjectionType;

protected:
   std::vector<TEveCaloData::vCellId_t*>   fCellLists;

public:
   TEveCalo2D(const Text_t* n="TEveCalo2D", const Text_t* t="");
   virtual ~TEveCalo2D(){}

   virtual void SetProjection(TEveProjectionManager* proj, TEveProjectable* model);
   virtual void UpdateProjection();
   virtual void SetDepth(Float_t x){fDepth = x;}

   virtual void ResetCache();

   virtual void ComputeBBox();

   ClassDef(TEveCalo2D, 0); // Class for visualization of projected calorimeter event data.
};
/**************************************************************************/
/**************************************************************************/

class TEveCaloLego : public TEveCaloViz
{
   friend class TEveCaloLegoGL;

public:
   enum EProjection_e { kAuto, k3D, k2D };
   enum E2DMode_e     { kValColor, kValSize };
   enum EBoxMode_e    { kNone, kFrontBack, kBack};

private:
   TEveCaloLego(const TEveCaloLego&);            // Not implemented
   TEveCaloLego& operator=(const TEveCaloLego&); // Not implemented

protected:
   TEveCaloData::vCellId_t fCellList;

   Color_t                 fFontColor;
   Color_t                 fGridColor;

   Int_t                   fNZStep; // Z axis label step in GeV
   Float_t                 fZAxisStep;

   Int_t                   fBinWidth; // distance in pixels of projected up and low edge

   EProjection_e           fProjection;
   E2DMode_e               f2DMode;
   EBoxMode_e              fBoxMode;

   Bool_t                  fDrawHPlane;
   Float_t                 fHPlaneVal;


public:
   TEveCaloLego(const Text_t* n="TEveCaloLego", const Text_t* t="");
   TEveCaloLego(TEveCaloData* data);

   virtual ~TEveCaloLego(){}

   Int_t  GetAxisStep(Float_t max) const;

   Color_t  GetFontColor() const { return fFontColor; }
   void     SetFontColor(Color_t ci) { fFontColor=ci; }

   Color_t  GetGridColor() const { return fGridColor; }
   void     SetGridColor(Color_t ci) { fGridColor=ci; }
  
   Int_t  GetNZStep() const { return fNZStep; }
   void   SetNZStep(Int_t s) { fNZStep = s;}
  
   Int_t    GetBinWidth() const { return fBinWidth; }
   void     SetBinWidth(Int_t bw) { fBinWidth = bw; }

   void           SetProjection(EProjection_e p) { fProjection = p; }
   EProjection_e  GetProjection() { return fProjection; }

   void       Set2DMode(E2DMode_e p) { f2DMode = p; }
   E2DMode_e  Get2DMode() { return f2DMode; }

   void       SetBoxMode(EBoxMode_e p) { fBoxMode = p; }
   EBoxMode_e  GetBoxMode() { return fBoxMode; }

   Bool_t GetDrawHPlane() const { return fDrawHPlane; }
   void   SetDrawHPlane(Bool_t s) { fDrawHPlane = s;}

   Float_t  GetHPlaneVal() const { return fHPlaneVal; }
   void     SetHPlaneVal(Float_t s) { fHPlaneVal = s;}

   virtual Float_t GetDefaultCellHeight() const;

   virtual void ResetCache();

   virtual void ComputeBBox();

   ClassDef(TEveCaloLego, 0);  // Class for visualization of calorimeter histogram data.
};

#endif
