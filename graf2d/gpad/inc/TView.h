// @(#)root/gpad:$Id$
// Author: Rene Brun, Nenad Buncic, Evgueni Tcherniaev, Olivier Couet   18/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TView
#define ROOT_TView


/////////////////////////////////////////////////////////////////////////
//                                                                     //
// TView  abstract interface for 3-D views                             //
//                                                                     //
/////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif

class TList;
class TSeqCollection;
class TVirtualPad;

class TView : public TObject, public TAttLine {

public:

   TView() {}
   TView(const TView &);
   virtual ~TView() {}

   virtual void          DefinePerspectiveView() = 0;
   virtual void          AxisVertex(Double_t ang, Double_t *av, Int_t &ix1, Int_t &ix2, Int_t &iy1, Int_t &iy2, Int_t &iz1, Int_t &iz2) = 0;
   virtual void          DefineViewDirection(const Double_t *s, const Double_t *c,
                                    Double_t cosphi, Double_t sinphi,
                                    Double_t costhe, Double_t sinthe,
                                    Double_t cospsi, Double_t sinpsi,
                                    Double_t *tnorm, Double_t *tback) = 0;
   virtual void          DrawOutlineCube(TList *outline, Double_t *rmin, Double_t *rmax) = 0;
   virtual void          ExecuteEvent(Int_t event, Int_t px, Int_t py) = 0;
   virtual void          ExecuteRotateView(Int_t event, Int_t px, Int_t py) = 0;
   virtual void          FindScope(Double_t *scale, Double_t *center, Int_t &irep) = 0;
   virtual Int_t         GetDistancetoAxis(Int_t axis, Int_t px, Int_t py, Double_t &ratio) = 0;
   virtual Double_t      GetDview() const = 0;
   virtual Double_t      GetDproj() const = 0;
   virtual Double_t      GetExtent() const = 0;
   virtual Bool_t        GetAutoRange() = 0;
   virtual Double_t      GetLatitude() = 0;
   virtual Double_t      GetLongitude() = 0;
   virtual Double_t      GetPsi() = 0;
   virtual void          GetRange (Float_t *min, Float_t *max) = 0;
   virtual void          GetRange (Double_t *min, Double_t *max) = 0;
   virtual Double_t     *GetRmax() = 0;
   virtual Double_t     *GetRmin() = 0;
   virtual TSeqCollection *GetOutline() = 0;
   virtual Double_t     *GetTback() = 0;
   virtual Double_t     *GetTN() = 0;
   virtual Double_t     *GetTnorm() = 0;
   virtual Int_t         GetSystem() = 0;
   virtual void          GetWindow(Double_t &u0, Double_t &v0, Double_t &du, Double_t &dv) const = 0;
   virtual Double_t      GetWindowWidth() const = 0;
   virtual Double_t      GetWindowHeight() const = 0;
   virtual void          FindNormal(Double_t x, Double_t  y, Double_t z, Double_t &zn) = 0;
   virtual void          FindPhiSectors(Int_t iopt, Int_t &kphi, Double_t *aphi, Int_t &iphi1, Int_t &iphi2) = 0;
   virtual void          FindThetaSectors(Int_t iopt, Double_t phi, Int_t &kth, Double_t *ath, Int_t &ith1, Int_t &ith2) = 0;
   virtual Bool_t        IsClippedNDC(Double_t *p) const = 0;
   virtual Bool_t        IsPerspective() const = 0;
   virtual Bool_t        IsViewChanged() const = 0;
   virtual void          NDCtoWC(const Float_t *pn, Float_t *pw) = 0;
   virtual void          NDCtoWC(const Double_t *pn, Double_t *pw) = 0;
   virtual void          NormalWCtoNDC(const Float_t *pw, Float_t *pn) = 0;
   virtual void          NormalWCtoNDC(const Double_t *pw, Double_t *pn) = 0;
   virtual void          PadRange(Int_t rback) = 0;
   virtual void          ResizePad() = 0;
   virtual void          SetAutoRange(Bool_t autorange=kTRUE) = 0;
   virtual void          SetAxisNDC(const Double_t *x1, const Double_t *x2, const Double_t *y1, const Double_t *y2, const Double_t *z1, const Double_t *z2) = 0;
   virtual void          SetDefaultWindow() = 0;
   virtual void          SetDview(Double_t dview) = 0;
   virtual void          SetDproj(Double_t dproj) = 0;
   virtual void          SetLatitude(Double_t latitude) = 0;
   virtual void          SetLongitude(Double_t longitude) = 0;
   virtual void          SetPsi(Double_t psi) = 0;
   virtual void          SetOutlineToCube() = 0;
   virtual void          SetParallel() = 0;
   virtual void          SetPerspective() = 0;
   virtual void          SetRange(const Double_t *min, const Double_t *max) = 0;
   virtual void          SetRange(Double_t x0, Double_t y0, Double_t z0, Double_t x1, Double_t y1, Double_t z1, Int_t flag=0) = 0;
   virtual void          SetSystem(Int_t system) = 0;
   virtual void          SetView(Double_t longitude, Double_t latitude, Double_t psi, Int_t &irep) = 0;
   virtual void          SetViewChanged(Bool_t flag=kTRUE) = 0;
   virtual void          SetWindow(Double_t u0, Double_t v0, Double_t du, Double_t dv) = 0;
   virtual void          WCtoNDC(const Float_t *pw, Float_t *pn) = 0;
   virtual void          WCtoNDC(const Double_t *pw, Double_t *pn) = 0;

//--
   virtual void          MoveFocus(Double_t *center, Double_t dx, Double_t dy, Double_t dz, Int_t nsteps=10,
                            Double_t dlong=0, Double_t dlat=0, Double_t dpsi=0) = 0;
   virtual void          MoveViewCommand(Char_t chCode, Int_t count=1) = 0;
   virtual void          MoveWindow(Char_t option) = 0;

   virtual void          AdjustScales(TVirtualPad *pad=0) = 0;
   virtual void          Centered3DImages(TVirtualPad *pad=0) = 0;
   virtual void          Centered() = 0;
   virtual void          FrontView(TVirtualPad *pad=0) = 0;
   virtual void          Front() = 0;

   virtual void          ZoomIn() = 0;
   virtual void          ZoomOut() = 0;
   virtual void          ZoomView(TVirtualPad *pad=0, Double_t zoomFactor = 1.25 ) = 0;
   virtual void          UnzoomView(TVirtualPad *pad=0,Double_t unZoomFactor = 1.25) = 0;

   virtual void          RotateView(Double_t phi, Double_t theta, TVirtualPad *pad=0) = 0;
   virtual void          SideView(TVirtualPad *pad=0) = 0;
   virtual void          Side() = 0;
   virtual void          TopView(TVirtualPad *pad=0) = 0;
   virtual void          Top() = 0;

   virtual void          ToggleRulers(TVirtualPad *pad=0) = 0;
   virtual void          ShowAxis() = 0;
   virtual void          ToggleZoom(TVirtualPad *pad=0) = 0;
   virtual void          ZoomMove() = 0;
   virtual void          Zoom() = 0;
   virtual void          UnZoom() = 0;

   static TView         *CreateView(Int_t system=1, const Double_t *rmin=0, const Double_t *rmax=0);

   ClassDef(TView,3);  //3-D View abstract interface for 3-D views
};

#endif

