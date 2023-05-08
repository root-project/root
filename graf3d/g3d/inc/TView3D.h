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
   virtual ~TView3D();

   virtual void     AxisVertex(Double_t ang, Double_t *av, Int_t &ix1, Int_t &ix2, Int_t &iy1, Int_t &iy2, Int_t &iz1, Int_t &iz2);
   virtual void     DefinePerspectiveView();
   virtual void     DefineViewDirection(const Double_t *s, const Double_t *c,
                                        Double_t cosphi, Double_t sinphi,
                                        Double_t costhe, Double_t sinthe,
                                        Double_t cospsi, Double_t sinpsi,
                                        Double_t *tnorm, Double_t *tback);
   virtual void      DrawOutlineCube(TList *outline, Double_t *rmin, Double_t *rmax);
   virtual void      ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void      ExecuteRotateView(Int_t event, Int_t px, Int_t py);
   virtual void      FindScope(Double_t *scale, Double_t *center, Int_t &irep);
   virtual Int_t     GetDistancetoAxis(Int_t axis, Int_t px, Int_t py, Double_t &ratio);
   virtual Double_t  GetDview() const {return fDview;}
   virtual Double_t  GetDproj() const {return fDproj;}
   virtual Double_t  GetExtent() const;
   virtual Bool_t    GetAutoRange() {return fAutoRange;}
   virtual Double_t  GetLatitude() {return fLatitude;}
   virtual Double_t  GetLongitude() {return fLongitude;}
   virtual Double_t  GetPsi() {return fPsi;}
   virtual void      GetRange (Float_t *min, Float_t *max);
   virtual void      GetRange (Double_t *min, Double_t *max);
   virtual Double_t *GetRmax() {return fRmax;}
   virtual Double_t *GetRmin() {return fRmin;}
   virtual TSeqCollection *GetOutline() {return fOutline; }
   virtual Double_t *GetTback() {return fTback;}
   virtual Double_t *GetTN() {return fTN;}
   virtual Double_t *GetTnorm() {return fTnorm;}
   virtual Int_t     GetSystem() {return fSystem;}
   virtual void      GetWindow(Double_t &u0, Double_t &v0, Double_t &du, Double_t &dv) const;
   virtual Double_t  GetWindowWidth() const {return 0.5*(fUVcoord[1]-fUVcoord[0]);}
   virtual Double_t  GetWindowHeight() const {return 0.5*(fUVcoord[3]-fUVcoord[2]);}
   virtual void      FindNormal(Double_t x, Double_t  y, Double_t z, Double_t &zn);
   virtual void      FindPhiSectors(Int_t iopt, Int_t &kphi, Double_t *aphi, Int_t &iphi1, Int_t &iphi2);
   virtual void      FindThetaSectors(Int_t iopt, Double_t phi, Int_t &kth, Double_t *ath, Int_t &ith1, Int_t &ith2);
   virtual Bool_t    IsClippedNDC(Double_t *p) const;
   virtual Bool_t    IsPerspective() const {return TestBit(kPerspective);}
   virtual Bool_t    IsViewChanged() const {return fChanged;}
   virtual void      NDCtoWC(const Float_t *pn, Float_t *pw);
   virtual void      NDCtoWC(const Double_t *pn, Double_t *pw);
   virtual void      NormalWCtoNDC(const Float_t *pw, Float_t *pn);
   virtual void      NormalWCtoNDC(const Double_t *pw, Double_t *pn);
   virtual void      PadRange(Int_t rback);
   virtual void      ResizePad();
   virtual void      SetAutoRange(Bool_t autorange=kTRUE) {fAutoRange=autorange;}
   virtual void      SetAxisNDC(const Double_t *x1, const Double_t *x2, const Double_t *y1, const Double_t *y2, const Double_t *z1, const Double_t *z2);
   virtual void      SetDefaultWindow();
   virtual void      SetDview(Double_t dview) {fDview=dview;}
   virtual void      SetDproj(Double_t dproj) {fDproj=dproj;}
   virtual void      SetLatitude(Double_t latitude) {fLatitude = latitude;}
   virtual void      SetLongitude(Double_t longitude) {fLongitude = longitude;}
   virtual void      SetPsi(Double_t psi) {fPsi = psi;}
   virtual void      SetOutlineToCube();
   virtual void      SetParallel(); // *MENU*
   virtual void      SetPerspective(); // *MENU*
   virtual void      SetRange(const Double_t *min, const Double_t *max);
   virtual void      SetRange(Double_t x0, Double_t y0, Double_t z0, Double_t x1, Double_t y1, Double_t z1, Int_t flag=0);
   virtual void      SetSystem(Int_t system) {fSystem = system;}
   virtual void      SetView(Double_t longitude, Double_t latitude, Double_t psi, Int_t &irep);
   virtual void      SetViewChanged(Bool_t flag=kTRUE) {fChanged = flag;}
   virtual void      SetWindow(Double_t u0, Double_t v0, Double_t du, Double_t dv);
   virtual void      WCtoNDC(const Float_t *pw, Float_t *pn);
   virtual void      WCtoNDC(const Double_t *pw, Double_t *pn);

//--
   virtual void      MoveFocus(Double_t *center, Double_t dx, Double_t dy, Double_t dz, Int_t nsteps=10,
                               Double_t dlong=0, Double_t dlat=0, Double_t dpsi=0);
   virtual void      MoveViewCommand(Char_t chCode, Int_t count=1);
   virtual void      MoveWindow(Char_t option);

   virtual void      AdjustScales(TVirtualPad *pad = nullptr);
   virtual void      Centered3DImages(TVirtualPad *pad = nullptr);
   virtual void      Centered();                       // *MENU*
   virtual void      FrontView(TVirtualPad *pad = nullptr);
   virtual void      Front();                          // *MENU*

   virtual void      ZoomIn(); // *MENU*
   virtual void      ZoomOut(); // *MENU*
   virtual void      ZoomView(TVirtualPad *pad = nullptr, Double_t zoomFactor = 1.25 );
   virtual void      UnzoomView(TVirtualPad *pad = nullptr,Double_t unZoomFactor = 1.25);

   virtual void      RotateView(Double_t phi, Double_t theta, TVirtualPad *pad = nullptr);
   virtual void      SideView(TVirtualPad *pad = nullptr);
   virtual void      Side();                          // *MENU*
   virtual void      TopView(TVirtualPad *pad = nullptr);
   virtual void      Top();                           // *MENU*

   virtual void      ToggleRulers(TVirtualPad *pad = nullptr);
   virtual void      ShowAxis();                      // *MENU*
   virtual void      ToggleZoom(TVirtualPad *pad = nullptr);
   virtual void      ZoomMove();                      // *MENU*
   virtual void      Zoom();                          // *MENU*
   virtual void      UnZoom();                        // *MENU*

   static  void      AdjustPad(TVirtualPad *pad = nullptr);

   ClassDef(TView3D,3);  //3-D View
};

#endif

