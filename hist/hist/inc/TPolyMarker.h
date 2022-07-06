// @(#)root/hist:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPolyMarker
#define ROOT_TPolyMarker


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPolyMarker                                                          //
//                                                                      //
// An array of points with the same marker.                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TObject.h"
#include "TAttMarker.h"
#include "TString.h"

class TCollection;

class TPolyMarker : public TObject, public TAttMarker {
protected:
   Int_t       fN{0};             ///< Number of points
   Int_t       fLastPoint{-1};    ///< The index of the last filled point
   Double_t   *fX{nullptr};       ///<[fN] Array of X coordinates
   Double_t   *fY{nullptr};       ///<[fN] Array of Y coordinates
   TString     fOption;           ///< Options

   TPolyMarker& operator=(const TPolyMarker&);

public:
   TPolyMarker();
   TPolyMarker(Int_t n, Option_t *option="");
   TPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   TPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   TPolyMarker(const TPolyMarker &polymarker);
   ~TPolyMarker() override;
   void     Copy(TObject &polymarker) const override;
   Int_t    DistancetoPrimitive(Int_t px, Int_t py) override;
   void     Draw(Option_t *option="") override;
   virtual void     DrawPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   void     ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   virtual Int_t    GetLastPoint() const { return fLastPoint;}
   virtual Int_t    GetN() const {return fN;}
   Option_t        *GetOption() const override {return fOption.Data();}
   Double_t        *GetX() const {return fX;}
   Double_t        *GetY() const {return fY;}
   void     ls(Option_t *option="") const override;
   virtual Int_t    Merge(TCollection *list);
   void     Paint(Option_t *option="") override;
   virtual void     PaintPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   void     Print(Option_t *option="") const override;
   void     SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual Int_t    SetNextPoint(Double_t x, Double_t y); // *MENU*
   virtual void     SetPoint(Int_t point, Double_t x, Double_t y); // *MENU*
   virtual void     SetPolyMarker(Int_t n);
   virtual void     SetPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   virtual void     SetPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   virtual Int_t    Size() const {return fLastPoint+1;}

   ClassDefOverride(TPolyMarker,4)  //An array of points with the same marker
};

#endif
