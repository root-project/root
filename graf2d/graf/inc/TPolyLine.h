// @(#)root/graf:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPolyLine
#define ROOT_TPolyLine


#include "TString.h"
#include "TObject.h"
#include "TAttLine.h"
#include "TAttFill.h"

class TCollection;

class TPolyLine : public TObject, public TAttLine, public TAttFill {

protected:
   Int_t        fN{0};            ///<Number of points
   Int_t        fLastPoint{-1};   ///<The index of the last filled point
   Double_t    *fX{nullptr};      ///<[fN] Array of X coordinates
   Double_t    *fY{nullptr};      ///<[fN] Array of Y coordinates
   TString      fOption;          ///<options

   TPolyLine& operator=(const TPolyLine&);

public:
   // TPolyLine status bits
   enum {
      kPolyLineNDC = BIT(14) ///< Polyline coordinates are in NDC space.
   };

   TPolyLine();
   TPolyLine(Int_t n, Option_t *option="");
   TPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   TPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   TPolyLine(const TPolyLine &polyline);
   virtual ~TPolyLine();

   void               Copy(TObject &polyline) const override;
   Int_t              DistancetoPrimitive(Int_t px, Int_t py) override;
   void               Draw(Option_t *option="") override;
   virtual TPolyLine *DrawPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   void               ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   virtual Int_t      GetLastPoint() const { return fLastPoint;}
   Int_t              GetN() const {return fN;}
   Double_t          *GetX() const {return fX;}
   Double_t          *GetY() const {return fY;}
   Option_t          *GetOption() const override { return fOption.Data(); }
   void               ls(Option_t *option="") const override;
   virtual Int_t      Merge(TCollection *list);
   void               Paint(Option_t *option="") override;
   virtual void       PaintPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   virtual void       PaintPolyLineNDC(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   void               Print(Option_t *option="") const override;
   void               SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void       SetNDC(Bool_t isNDC=kTRUE);
   virtual Int_t      SetNextPoint(Double_t x, Double_t y); // *MENU*
   virtual void       SetOption(Option_t *option="") {fOption = option;}
   virtual void       SetPoint(Int_t point, Double_t x, Double_t y); // *MENU*
   virtual void       SetPolyLine(Int_t n);
   virtual void       SetPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   virtual void       SetPolyLine(Int_t n, Double_t *x, Double_t *y3, Option_t *option="");
   virtual Int_t      Size() const {return fLastPoint+1;}

   ClassDefOverride(TPolyLine,3)  //A PolyLine
};

#endif

