// @(#)root/base:$Name:  $:$Id: TView.h,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
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
// TView                                                               //
//                                                                     //
/////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif

class TSeqCollection;
class TVirtualPad;

class TView : public TObject, public TAttLine {


protected:
        Double_t        fLatitude;         //View angle latitude
        Double_t        fLongitude;        //View angle longitude
        Double_t        fPsi;              //View angle psi
        Double_t        fTN[12];           //
        Double_t        fTB[12];           //
        Double_t        fRmax[3];          //Upper limits of object
        Double_t        fRmin[3];          //Lower limits of object
        Double_t        fTnorm[12];        //Transformation matrix
        Double_t        fTback[12];        //Back transformation matrix
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
        void            ResetView(Double_t longitude, Double_t latitude, Double_t psi, Int_t &irep);


public:
                TView();
                TView(Int_t system);
                TView(Float_t *rmin, Float_t *rmax, Int_t system = 1);
                TView(Double_t *rmin, Double_t *rmax, Int_t system = 1);
virtual         ~TView();
   virtual void AxisVertex(Double_t ang, Double_t *av, Int_t &ix1, Int_t &ix2, Int_t &iy1, Int_t &iy2, Int_t &iz1, Int_t &iz2);
   virtual void DefineViewDirection(Double_t *s, Double_t *c,
                                    Double_t cosphi, Double_t sinphi,
                                    Double_t costhe, Double_t sinthe,
                                    Double_t cospsi, Double_t sinpsi,
                                    Double_t *tnorm, Double_t *tback);
   virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void  ExecuteRotateView(Int_t event, Int_t px, Int_t py);
   virtual void  FindScope(Double_t *scale, Double_t *center, Int_t &irep);
   virtual Int_t GetDistancetoAxis(Int_t axis, Int_t px, Int_t py, Double_t &ratio);
Bool_t           GetAutoRange() {return fAutoRange;}
Double_t         GetLatitude() {return fLatitude;}
Double_t         GetLongitude() {return fLongitude;}
Double_t         GetPsi() {return fPsi;}
   virtual void  GetRange (Float_t *min, Float_t *max);
   virtual void  GetRange (Double_t *min, Double_t *max);
Double_t        *GetRmax() {return fRmax;}
Double_t        *GetRmin() {return fRmin;}
TSeqCollection  *GetOutline() {return fOutline; }
Double_t        *GetTN() {return fTN;}
Double_t        *GetTnorm() {return fTnorm;}
  Int_t          GetSystem() {return fSystem;}
   virtual void  FindNormal(Double_t x, Double_t  y, Double_t z, Double_t &zn);
   virtual void  FindPhiSectors(Int_t iopt, Int_t &kphi, Double_t *aphi, Int_t &iphi1, Int_t &iphi2);
   virtual void  FindThetaSectors(Int_t iopt, Double_t phi, Int_t &kth, Double_t *ath, Int_t &ith1, Int_t &ith2);
   virtual void  NDCtoWC(Float_t *pn, Float_t *pw);
   virtual void  NDCtoWC(Double_t *pn, Double_t *pw);
   virtual void  NormalWCtoNDC(Float_t *pw, Float_t *pn);
   virtual void  NormalWCtoNDC(Double_t *pw, Double_t *pn);
   virtual void  PadRange(Double_t rback);
   virtual void  SetAutoRange(Bool_t autorange=kTRUE) {fAutoRange=autorange;}
   virtual void  SetAxisNDC(Double_t *x1, Double_t *x2, Double_t *y1, Double_t *y2, Double_t *z1, Double_t *z2);
   void          SetLatitude(Double_t latitude) {fLatitude = latitude;}
   void          SetLongitude(Double_t longitude) {fLongitude = longitude;}
   void          SetPsi(Double_t psi) {fPsi = psi;}
   virtual void  SetOutlineToCube();
   virtual void  SetRange(Double_t *min, Double_t *max);
   virtual void  SetRange(Double_t x0, Double_t y0, Double_t z0, Double_t x1, Double_t y1, Double_t z1, Int_t flag=0);
   virtual void  SetSystem(Int_t system) {fSystem = system;}
   virtual void  SetView(Double_t longitude, Double_t latitude, Double_t psi, Int_t &irep);
   virtual void  WCtoNDC(Float_t *pw, Float_t *pn);
   virtual void  WCtoNDC(Double_t *pw, Double_t *pn);

//--
    virtual void MoveViewCommand(Char_t chCode, Int_t count=1);

    static  void AdjustPad(TVirtualPad *pad=0);
    virtual void AdjustScales(TVirtualPad *pad=0); // *MENU*
    virtual void Centered3DImages(TVirtualPad *pad=0);
            void Centered();                       // *MENU*
    virtual void FrontView(TVirtualPad *pad=0);
            void Front();                          // *MENU*

    virtual void ZoomView(TVirtualPad *pad=0, Double_t zoomFactor = 1.25 );
    virtual void UnzoomView(TVirtualPad *pad=0,Double_t unZoomFactor = 1.25);

    virtual void RotateView(Double_t phi, Double_t theta, TVirtualPad *pad=0);
    virtual void SideView(TVirtualPad *pad=0);
            void Side();                          // *MENU*
    virtual void TopView(TVirtualPad *pad=0);
            void Top();                           // *MENU*

    virtual void ToggleRulers(TVirtualPad *pad=0);
            void ShowAxis();                      // *MENU*
    virtual void ToggleZoom(TVirtualPad *pad=0);
            void ZoomMove();                      // *MENU*

   ClassDef(TView,1)  //3-D View
};

//      Shortcuts for menus
inline void TView::Centered(){Centered3DImages();}
inline void TView::Front()   {FrontView();}
inline void TView::ShowAxis(){ToggleRulers(); }
inline void TView::Side()    {SideView();}
inline void TView::Top()     {TopView();}
inline void TView::ZoomMove(){ToggleZoom();}

#endif

