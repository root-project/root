// @(#)root/graf:$Id$
// Author: Rene Brun   12/05/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMarker
#define ROOT_TMarker


#include "TObject.h"
#include "TAttMarker.h"
#include "TAttBBox2D.h"

class TPoint;

class TMarker : public TObject, public TAttMarker, public TAttBBox2D {

protected:
   Double_t     fX;           ///< X position of marker (left,center,etc..)
   Double_t     fY;           ///< Y position of marker (left,center,etc..)

public:
   // TMarker status bits
   enum {
      kMarkerNDC = BIT(14)  ///< Marker position is in NDC
   };

   TMarker();
   TMarker(Double_t x, Double_t y, Int_t marker);
   TMarker(const TMarker &marker);
   virtual ~TMarker();

   void             Copy(TObject &marker) const override;
   Int_t            DistancetoPrimitive(Int_t px, Int_t py) override;
   void             Draw(Option_t *option="") override;
   virtual TMarker *DrawMarker(Double_t x, Double_t y);
   void             ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   Double_t         GetX() const  {return fX;}
   Double_t         GetY() const  {return fY;}
   void             ls(Option_t *option="") const override;
   void             Paint(Option_t *option="") override;
   virtual void     PaintMarker(Double_t x, Double_t y);
   virtual void     PaintMarkerNDC(Double_t u, Double_t v);
   void             Print(Option_t *option="") const override;
   void             SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void     SetNDC(Bool_t isNDC=kTRUE);
   virtual void     SetX(Double_t x) { fX = x;} // *MENU*
   virtual void     SetY(Double_t y) { fY = y;} // *MENU*

   Rectangle_t      GetBBox() override;
   TPoint           GetBBoxCenter() override;
   void             SetBBoxCenter(const TPoint &p) override;
   void             SetBBoxCenterX(const Int_t x) override;
   void             SetBBoxCenterY(const Int_t y) override;
   void             SetBBoxX1(const Int_t x) override;
   void             SetBBoxX2(const Int_t x) override;
   void             SetBBoxY1(const Int_t y) override;
   void             SetBBoxY2(const Int_t y) override;

   static  void     DisplayMarkerTypes();
   static  void     DisplayMarkerLineWidths();

   ClassDefOverride(TMarker,3)  //Marker
};

#endif
