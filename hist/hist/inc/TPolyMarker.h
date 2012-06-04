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


#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TCollection;

class TPolyMarker : public TObject, public TAttMarker {
protected:
   Int_t       fN;            //number of points
   Int_t       fLastPoint;    //The index of the last filled point
   Double_t   *fX;            //[fN] Array of X coordinates
   Double_t   *fY;            //[fN] Array of Y coordinates
   TString     fOption;       //options

   TPolyMarker& operator=(const TPolyMarker&);

public:
   TPolyMarker();
   TPolyMarker(Int_t n, Option_t *option="");
   TPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   TPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   TPolyMarker(const TPolyMarker &polymarker);
   virtual ~TPolyMarker();
   virtual void     Copy(TObject &polymarker) const;
   virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   virtual void     Draw(Option_t *option="");
   virtual void     DrawPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual Int_t    GetLastPoint() const { return fLastPoint;}
   virtual Int_t    GetN() const {return fN;}
   Option_t        *GetOption() const {return fOption.Data();}
   Double_t        *GetX() const {return fX;}
   Double_t        *GetY() const {return fY;}
   virtual void     ls(Option_t *option="") const;
   virtual Int_t    Merge(TCollection *list);
   virtual void     Paint(Option_t *option="");
   virtual void     PaintPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   virtual void     Print(Option_t *option="") const;
   virtual void     SavePrimitive(std::ostream &out, Option_t *option = "");
   virtual Int_t    SetNextPoint(Double_t x, Double_t y); // *MENU*
   virtual void     SetPoint(Int_t point, Double_t x, Double_t y); // *MENU*
   virtual void     SetPolyMarker(Int_t n);
   virtual void     SetPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   virtual void     SetPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   virtual Int_t    Size() const {return fLastPoint+1;}

   ClassDef(TPolyMarker,4)  //An array of points with the same marker
};

#endif
