// @(#)root/graf:$Id$
// Author: Rene Brun   16/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPave
#define ROOT_TPave


#include "TBox.h"
#include "TString.h"

class TPave : public TBox {

protected:
   Double_t     fX1NDC;         ///< X1 point in NDC coordinates
   Double_t     fY1NDC;         ///< Y1 point in NDC coordinates
   Double_t     fX2NDC;         ///< X2 point in NDC coordinates
   Double_t     fY2NDC;         ///< Y2 point in NDC coordinates
   Int_t        fBorderSize;    ///< window box bordersize in pixels
   Int_t        fInit;          ///< (=0 if transformation to NDC not yet done)
   Int_t        fShadowColor;   ///< Color of the pave's shadow
   Double_t     fCornerRadius;  ///< Corner radius in case of option arc
   TString      fOption;        ///< Pave style
   TString      fName;          ///< Pave name

public:
   // TPave status bits
   enum {
      kNameIsAction = BIT(11)   ///< double clicking on TPave will execute action
   };

   TPave();
   TPave(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
         Int_t bordersize=4 ,Option_t *option="br");
   TPave(const TPave &pave);
   virtual ~TPave();

   TPave &operator=(const TPave &src);

   void           Copy(TObject &pave) const override;
   virtual void   ConvertNDCtoPad();
   Int_t          DistancetoPrimitive(Int_t px, Int_t py) override;
   void           Draw(Option_t *option="") override;
   virtual TPave *DrawPave(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                          Int_t bordersize=4 ,Option_t *option="br");
   void           ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   Int_t          GetBorderSize() const { return fBorderSize;}
   Double_t       GetCornerRadius() const {return fCornerRadius;}
   Option_t      *GetName() const override {return fName.Data();}
   Option_t      *GetOption() const override {return fOption.Data();}
   Int_t          GetShadowColor() const {return fShadowColor;}
   Double_t       GetX1NDC() const {return fX1NDC;}
   Double_t       GetX2NDC() const {return fX2NDC;}
   Double_t       GetY1NDC() const {return fY1NDC;}
   Double_t       GetY2NDC() const {return fY2NDC;}
   ULong_t        Hash() const override { return fName.Hash(); }
   Bool_t         IsSortable() const override { return kTRUE; }
   void           ls(Option_t *option="") const override;
   void           Paint(Option_t *option="") override;
   virtual void   PaintPave(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                            Int_t bordersize=4 ,Option_t *option="br");
   virtual void   PaintPaveArc(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                               Int_t bordersize=4 ,Option_t *option="br");
   void           Print(Option_t *option="") const override;
   void           SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void   SetBorderSize(Int_t bordersize=4) {fBorderSize = bordersize;} // *MENU*
   virtual void   SetCornerRadius(Double_t rad = 0.2) {fCornerRadius = rad;} // *MENU*
   virtual void   SetName(const char *name="") {fName = name;} // *MENU*
   virtual void   SetOption(Option_t *option="br") {fOption = option;}
   virtual void   SetShadowColor(Int_t color) {fShadowColor=color;} // *MENU*
   virtual void   SetX1NDC(Double_t x1) {fX1NDC=x1;}
   virtual void   SetX2NDC(Double_t x2) {fX2NDC=x2;}
   virtual void   SetY1NDC(Double_t y1) {fY1NDC=y1;}
   virtual void   SetY2NDC(Double_t y2) {fY2NDC=y2;}
   virtual void   SetX1(Double_t x1) override;
   virtual void   SetX2(Double_t x2) override;
   virtual void   SetY1(Double_t y1) override;
   virtual void   SetY2(Double_t y2) override;

   ClassDefOverride(TPave,3)  //Pave. A box with shadowing
};

#endif

