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
   TEveCaloData* fData;           // event data reference
   Bool_t        fCellIdCacheOK;  // data cell ids cache state

   Double_t      fEtaMin;
   Double_t      fEtaMax;

   Double_t      fPhi;
   Double_t      fPhiOffset;     // phi range +/- offset

   Bool_t        fAutoRange;     // set eta phi limits on DataChanged()

   Float_t       fBarrelRadius;  // barrel raidus in cm
   Float_t       fEndCapPos;     // end cap z coordinate in cm

   Float_t       fPlotEt;        // plot E or Et.

   Float_t           fMaxTowerH;  // bounding box z dimesion
   Bool_t            fScaleAbs;
   Float_t           fMaxValAbs;

   Bool_t            fValueIsColor;   // Interpret signal value as RGBA color.
   TEveRGBAPalette*  fPalette;        // Pointer to signal-color palette.


   void AssignCaloVizParameters(TEveCaloViz* cv);

   void SetupColorHeight(Float_t value, Int_t slice, Float_t& height) const;

   virtual void BuildCellIdCache() = 0;

public:
   TEveCaloViz(TEveCaloData* data=0, const char* n="TEveCaloViz", const char* t="");

   virtual ~TEveCaloViz();

   virtual void IncImpliedSelected();

   virtual TEveElement* ForwardSelection() const;

   virtual void Paint(Option_t* option="");

   virtual TClass* ProjectedClass() const;

   TEveCaloData* GetData() const { return fData; }
   void    SetData(TEveCaloData* d);
   void    DataChanged();
   Float_t GetMaxVal() const;
   void    InvalidateCellIdCache() { fCellIdCacheOK=kFALSE; ResetBBox(); };

   Float_t GetDataSliceThreshold(Int_t slice) const;
   void    SetDataSliceThreshold(Int_t slice, Float_t val);
   Color_t GetDataSliceColor(Int_t slice) const;
   void    SetDataSliceColor(Int_t slice, Color_t col);

   Float_t GetBarrelRadius() const { return fBarrelRadius; }
   void    SetBarrelRadius(Float_t r) { fBarrelRadius = r; ResetBBox(); }
   Float_t GetEndCapPos   () const { return fEndCapPos; }
   void    SetEndCapPos   (Float_t z) { fEndCapPos = z; ResetBBox(); }

   Bool_t  GetPlotEt() const { return fPlotEt; }
   void    SetPlotEt(Bool_t x);

   void    SetMaxTowerH(Float_t x) { fMaxTowerH = x; }
   Float_t GetMaxTowerH() const    { return fMaxTowerH; }
   void    SetScaleAbs(Bool_t x) { fScaleAbs = x; }
   Bool_t  GetScaleAbs() const { return fScaleAbs; }
   void    SetMaxValAbs(Float_t x) { fMaxValAbs = x; }
   Float_t GetMaxValAbs() const    { return fMaxValAbs; }

   Float_t GetTransitionEta() const;
   Float_t GetTransitionTheta() const;

   TEveRGBAPalette* GetPalette() const { return fPalette; }
   void             SetPalette(TEveRGBAPalette* p);

   TEveRGBAPalette* AssertPalette();
   Bool_t  GetValueIsColor()   const { return fValueIsColor;}
   void    SetValueIsColor(Bool_t x) { fValueIsColor = x;}

   Float_t GetValToHeight() const;
   Bool_t  GetAutoRange()   const { return fAutoRange; }
   void    SetAutoRange(Bool_t x) { fAutoRange = x; }

   void    SetEta(Float_t l, Float_t u);
   Float_t GetEta()    const { return 0.5f*(fEtaMin+fEtaMax); }
   Float_t GetEtaMin() const { return fEtaMin; }
   Float_t GetEtaMax() const { return fEtaMax; }
   Float_t GetEtaRng() const { return fEtaMax-fEtaMin; }

   void    SetPhi(Float_t phi)    { SetPhiWithRng(phi, fPhiOffset); }
   void    SetPhiRng(Float_t rng) { SetPhiWithRng(fPhi, rng); }
   void    SetPhiWithRng(Float_t x, Float_t r);
   Float_t GetPhi()    const { return fPhi; }
   Float_t GetPhiMin() const { return fPhi-fPhiOffset; }
   Float_t GetPhiMax() const { return fPhi+fPhiOffset; }
   Float_t GetPhiRng() const { return 2.0f*fPhiOffset; }


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
   TEveCaloData::vCellId_t fCellListSelected;

   Bool_t    fRnrEndCapFrame;
   Bool_t    fRnrBarrelFrame;

   Color_t   fFrameColor;
   UChar_t   fFrameTransparency;

   virtual void BuildCellIdCache();

public:
   TEveCalo3D(TEveCaloData* d=0, const char* n="TEveCalo3D", const char* t="xx");
   virtual ~TEveCalo3D() {}
   virtual void ComputeBBox();

   virtual Bool_t CanEditMainColor()        const { return kTRUE; }
   virtual Bool_t CanEditMainTransparency() const { return kTRUE; }

   void SetRnrFrame(Bool_t e, Bool_t b)         { fRnrEndCapFrame = e; fRnrBarrelFrame = b; }
   void GetRnrFrame(Bool_t &e, Bool_t &b) const { e = fRnrEndCapFrame; b = fRnrBarrelFrame; }

   void    SetFrameTransparency(UChar_t x) { fFrameTransparency = x; }
   UChar_t GetFrameTransparency() const { return fFrameTransparency; }

   ClassDef(TEveCalo3D, 0); // Class for 3D visualization of calorimeter event data.
};

/**************************************************************************/
/**************************************************************************/

class TEveCalo2D : public TEveCaloViz,
                   public TEveProjected
{
   friend class TEveCalo2DGL;

   typedef std::vector<TEveCaloData::vCellId_t*>           vBinCells_t;
   typedef std::vector<TEveCaloData::vCellId_t*>::iterator vBinCells_i;

private:
   TEveCalo2D(const TEveCalo2D&);            // Not implemented
   TEveCalo2D& operator=(const TEveCalo2D&); // Not implemented

   TEveProjection::EPType_e  fOldProjectionType;

protected:
   std::vector<TEveCaloData::vCellId_t*>   fCellLists;

   std::vector<TEveCaloData::vCellId_t*>   fCellListsSelected;
   std::vector<Int_t>                      fBinIdsSelected;

   virtual void BuildCellIdCache();
   virtual void BuildCellIdCacheSelected();

public:
   TEveCalo2D(const char* n="TEveCalo2D", const char* t="");
   virtual ~TEveCalo2D();

   virtual void SetProjection(TEveProjectionManager* proj, TEveProjectable* model);
   virtual void UpdateProjection();
   virtual void SetDepth(Float_t x){fDepth = x;}

   virtual void ComputeBBox();

   ClassDef(TEveCalo2D, 0); // Class for visualization of projected calorimeter event data.
};
/**************************************************************************/
/**************************************************************************/

class TEveCaloLego : public TEveCaloViz
{
   friend class TEveCaloLegoGL;
   friend class TEveCaloLegoOverlay;

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
   Color_t                 fPlaneColor;
   UChar_t                 fPlaneTransparency;

   Int_t                   fNZSteps; // Z axis label step in GeV
   Float_t                 fZAxisStep;

   Bool_t                  fAutoRebin;
   Int_t                   fPixelsPerBin;
   Bool_t                  fNormalizeRebin;

   EProjection_e           fProjection;
   E2DMode_e               f2DMode;
   EBoxMode_e              fBoxMode;  // additional scale info

   Bool_t                  fDrawHPlane;
   Float_t                 fHPlaneVal;

   Int_t                   fBinStep;

   Int_t                   fDrawNumberCellPixels;
   Int_t                   fCellPixelFontSize;

   virtual void BuildCellIdCache();

public:
   TEveCaloLego(TEveCaloData* data=0, const char* n="TEveCaloLego", const char* t="");
   virtual ~TEveCaloLego(){}

   virtual void ComputeBBox();
   virtual void  SetData(TEveCaloData* d);

   Color_t  GetFontColor() const { return fFontColor; }
   void     SetFontColor(Color_t ci) { fFontColor=ci; }

   Color_t  GetGridColor() const { return fGridColor; }
   void     SetGridColor(Color_t ci) { fGridColor=ci; }

   Color_t  GetPlaneColor() const { return fPlaneColor; }
   void     SetPlaneColor(Color_t ci) { fPlaneColor=ci; }

   UChar_t  GetPlaneTransparency() const { return fPlaneTransparency; }
   void     SetPlaneTransparency(UChar_t t) { fPlaneTransparency=t; }

   Int_t    GetNZSteps() const { return fNZSteps; }
   void     SetNZSteps(Int_t s) { fNZSteps = s;}

   Int_t    GetPixelsPerBin() const { return fPixelsPerBin; }
   void     SetPixelsPerBin(Int_t bw) { fPixelsPerBin = bw; }

   Bool_t   GetAutoRebin() const { return fAutoRebin; }
   void     SetAutoRebin(Bool_t s) { fAutoRebin = s;}

   Bool_t   GetNormalizeRebin() const { return fNormalizeRebin; }
   void     SetNormalizeRebin(Bool_t s) { fNormalizeRebin = s; fCellIdCacheOK=kFALSE;}

   void           SetProjection(EProjection_e p) { fProjection = p; }
   EProjection_e  GetProjection() { return fProjection; }

   void       Set2DMode(E2DMode_e p) { f2DMode = p; }
   E2DMode_e  Get2DMode() { return f2DMode; }

   void        SetBoxMode(EBoxMode_e p) { fBoxMode = p; }
   EBoxMode_e  GetBoxMode() { return fBoxMode; }

   Bool_t   GetDrawHPlane() const { return fDrawHPlane; }
   void     SetDrawHPlane(Bool_t s) { fDrawHPlane = s;}

   Float_t  GetHPlaneVal() const { return fHPlaneVal; }
   void     SetHPlaneVal(Float_t s) { fHPlaneVal = s;}

   Int_t    GetDrawNumberCellPixels() { return fDrawNumberCellPixels; }
   void     SetDrawNumberCellPixels(Int_t x) { fDrawNumberCellPixels = x; }
   Int_t    GetCellPixelFontSize() { return fCellPixelFontSize; }
   void     SetCellPixelFontSize(Int_t x) { fCellPixelFontSize = x; }

   ClassDef(TEveCaloLego, 0);  // Class for visualization of calorimeter histogram data.
};

#endif
