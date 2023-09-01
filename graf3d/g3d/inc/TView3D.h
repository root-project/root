// @(#)root/g3d:$Id$
// Author: Rene Brun 19/02/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TView3D
#define ROOT_TView3D


/////////////////////////////////////////////////////////////////////////
//                                                                     //
// TView3D                                                             //
//                                                                     //
/////////////////////////////////////////////////////////////////////////


#include "TView.h"

class TSeqCollection;
class TVirtualPad;

class TView3D : public TView {

protected:
   Double_t        fLatitude;         //View angle latitude
   Double_t        fLongitude;        //View angle longitude
   Double_t        fPsi;              //View angle psi
   Double_t        fDview;            //Distance from COP to COV
   Double_t        fDproj;            //Distance from COP to projection plane
   Double_t        fUpix;             // pad X size in pixels
   Double_t        fVpix;             // pad Y size in pixels
   Double_t        fTN[16];           //
   Double_t        fTB[16];           //
   Double_t        fRmax[3];          //Upper limits of object
   Double_t        fRmin[3];          //Lower limits of object
   Double_t        fUVcoord[4];       //Viewing window limits
   Double_t        fTnorm[16];        //Transformation matrix
   Double_t        fTback[16];        //Back transformation matrix
   Double_t        fX1[3];            //First coordinate of X axis
   Double_t        fX2[3];            //Second coordinate of X axis
   Double_t        fY1[3];            //First coordinate of Y axis
   Double_t        fY2[3];            //Second coordinate of Y axis
   Double_t        fZ1[3];            //First coordinate of Z axis
   Double_t        fZ2[3];            //Second coordinate of Z axis
   Int_t           fSystem;           //Coordinate system
   TSeqCollection *fOutline;          //Collection of outline's objects
   Bool_t          fDefaultOutline;   //Set to TRUE if outline is default cube
   Bool_t          fAutoRange;        //Set to TRUE if range computed automatically
   Bool_t          fChanged;          //! Set to TRUE after ExecuteRotateView

   TView3D(const TView3D&); // Not implemented
   TView3D& operator=(const TView3D&); // Not implemented

   void            ResetView(Double_t longitude, Double_t latitude, Double_t psi, Int_t &irep);


public:
   // TView3D status bits
   enum {
      kPerspective  = BIT(6)
   };

   TView3D();
   TView3D(Int_t system, const Double_t *rmin, const Double_t *rmax);
   ~TView3D() override;

   void     AxisVertex(Double_t ang, Double_t *av, Int_t &ix1, Int_t &ix2, Int_t &iy1, Int_t &iy2, Int_t &iz1, Int_t &iz2) override;
   void     DefinePerspectiveView() override;
   void     DefineViewDirection(const Double_t *s, const Double_t *c,
                                        Double_t cosphi, Double_t sinphi,
                                        Double_t costhe, Double_t sinthe,
                                        Double_t cospsi, Double_t sinpsi,
                                        Double_t *tnorm, Double_t *tback) override;
   void      DrawOutlineCube(TList *outline, Double_t *rmin, Double_t *rmax) override;
   void      ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   void      ExecuteRotateView(Int_t event, Int_t px, Int_t py) override;
   void      FindScope(Double_t *scale, Double_t *center, Int_t &irep) override;
   Int_t     GetDistancetoAxis(Int_t axis, Int_t px, Int_t py, Double_t &ratio) override;
   Double_t  GetDview() const override {return fDview;}
   Double_t  GetDproj() const override {return fDproj;}
   Double_t  GetExtent() const override;
   Bool_t    GetAutoRange() override {return fAutoRange;}
   Double_t  GetLatitude() override {return fLatitude;}
   Double_t  GetLongitude() override {return fLongitude;}
   Double_t  GetPsi() override {return fPsi;}
   void      GetRange (Float_t *min, Float_t *max) override;
   void      GetRange (Double_t *min, Double_t *max) override;
   Double_t *GetRmax() override {return fRmax;}
   Double_t *GetRmin() override {return fRmin;}
   TSeqCollection *GetOutline() override {return fOutline; }
   Double_t *GetTback() override {return fTback;}
   Double_t *GetTN() override {return fTN;}
   Double_t *GetTnorm() override {return fTnorm;}
   Int_t     GetSystem() override {return fSystem;}
   void      GetWindow(Double_t &u0, Double_t &v0, Double_t &du, Double_t &dv) const override;
   Double_t  GetWindowWidth() const override {return 0.5*(fUVcoord[1]-fUVcoord[0]);}
   Double_t  GetWindowHeight() const override {return 0.5*(fUVcoord[3]-fUVcoord[2]);}
   void      FindNormal(Double_t x, Double_t  y, Double_t z, Double_t &zn) override;
   void      FindPhiSectors(Int_t iopt, Int_t &kphi, Double_t *aphi, Int_t &iphi1, Int_t &iphi2) override;
   void      FindThetaSectors(Int_t iopt, Double_t phi, Int_t &kth, Double_t *ath, Int_t &ith1, Int_t &ith2) override;
   Bool_t    IsClippedNDC(Double_t *p) const override;
   Bool_t    IsPerspective() const override {return TestBit(kPerspective);}
   Bool_t    IsViewChanged() const override {return fChanged;}
   void      NDCtoWC(const Float_t *pn, Float_t *pw) override;
   void      NDCtoWC(const Double_t *pn, Double_t *pw) override;
   void      NormalWCtoNDC(const Float_t *pw, Float_t *pn) override;
   void      NormalWCtoNDC(const Double_t *pw, Double_t *pn) override;
   void      PadRange(Int_t rback) override;
   void      ResizePad() override;
   void      SetAutoRange(Bool_t autorange=kTRUE) override {fAutoRange=autorange;}
   void      SetAxisNDC(const Double_t *x1, const Double_t *x2, const Double_t *y1, const Double_t *y2, const Double_t *z1, const Double_t *z2) override;
   void      SetDefaultWindow() override;
   void      SetDview(Double_t dview) override {fDview=dview;}
   void      SetDproj(Double_t dproj) override {fDproj=dproj;}
   void      SetLatitude(Double_t latitude) override {fLatitude = latitude;}
   void      SetLongitude(Double_t longitude) override {fLongitude = longitude;}
   void      SetPsi(Double_t psi) override {fPsi = psi;}
   void      SetOutlineToCube() override;
   void      SetParallel() override; // *MENU*
   void      SetPerspective() override; // *MENU*
   void      SetRange(const Double_t *min, const Double_t *max) override;
   void      SetRange(Double_t x0, Double_t y0, Double_t z0, Double_t x1, Double_t y1, Double_t z1, Int_t flag=0) override;
   void      SetSystem(Int_t system) override {fSystem = system;}
   void      SetView(Double_t longitude, Double_t latitude, Double_t psi, Int_t &irep) override;
   void      SetViewChanged(Bool_t flag=kTRUE) override {fChanged = flag;}
   void      SetWindow(Double_t u0, Double_t v0, Double_t du, Double_t dv) override;
   void      WCtoNDC(const Float_t *pw, Float_t *pn) override;
   void      WCtoNDC(const Double_t *pw, Double_t *pn) override;

//--
   void      MoveFocus(Double_t *center, Double_t dx, Double_t dy, Double_t dz, Int_t nsteps=10,
                               Double_t dlong=0, Double_t dlat=0, Double_t dpsi=0) override;
   void      MoveViewCommand(Char_t chCode, Int_t count=1) override;
   void      MoveWindow(Char_t option) override;

   void      AdjustScales(TVirtualPad *pad = nullptr) override;
   void      Centered3DImages(TVirtualPad *pad = nullptr) override;
   void      Centered() override;                       // *MENU*
   void      FrontView(TVirtualPad *pad = nullptr) override;
   void      Front() override;                          // *MENU*

   void      ZoomIn() override; // *MENU*
   void      ZoomOut() override; // *MENU*
   void      ZoomView(TVirtualPad *pad = nullptr, Double_t zoomFactor = 1.25 ) override;
   void      UnzoomView(TVirtualPad *pad = nullptr,Double_t unZoomFactor = 1.25) override;

   void      RotateView(Double_t phi, Double_t theta, TVirtualPad *pad = nullptr) override;
   void      SideView(TVirtualPad *pad = nullptr) override;
   void      Side() override;                          // *MENU*
   void      TopView(TVirtualPad *pad = nullptr) override;
   void      Top() override;                           // *MENU*

   void      ToggleRulers(TVirtualPad *pad = nullptr) override;
   void      ShowAxis() override;                      // *MENU*
   void      ToggleZoom(TVirtualPad *pad = nullptr) override;
   void      ZoomMove() override;                      // *MENU*
   void      Zoom() override;                          // *MENU*
   void      UnZoom() override;                        // *MENU*

   static  void      AdjustPad(TVirtualPad *pad = nullptr);

   ClassDefOverride(TView3D,3);  //3-D View
};

#endif

