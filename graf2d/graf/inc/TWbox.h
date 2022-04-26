// @(#)root/graf:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWbox
#define ROOT_TWbox


#include "TBox.h"

#include "TColor.h"

class TWbox : public TBox {

protected:
   Short_t      fBorderSize{0};    ///< window box bordersize in pixels
   Short_t      fBorderMode{0};    ///< Bordermode (-1=down, 0 = no border, 1=up)

public:
   TWbox() {} // NOLINT: not allowed to use = default because of TObject::kIsOnHeap detection, see ROOT-10300
   TWbox(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
         Color_t color=18, Short_t bordersize=5 ,Short_t bordermode=1);
   TWbox(const TWbox &wbox);
   virtual ~TWbox() = default;

   TWbox &operator=(const TWbox &src);

   void           Copy(TObject &wbox) const override;
   void           Draw(Option_t *option="") override;
   virtual TWbox *DrawWbox(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                           Color_t color=33 ,Short_t bordersize=5 ,Short_t bordermode=-1);
   void           ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   Short_t        GetBorderMode() const { return fBorderMode;}
   Short_t        GetBorderSize() const { return fBorderSize;}
   Int_t          GetDarkColor() const  {return TColor::GetColorDark(GetFillColor());}
   Int_t          GetLightColor() const {return TColor::GetColorBright(GetFillColor());}
   void           Paint(Option_t *option="") override;
   virtual void   PaintFrame(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                             Color_t color, Short_t bordersize, Short_t bordermode,
                             Bool_t tops);
   virtual void   PaintWbox(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                  Color_t color=33, Short_t bordersize=5, Short_t bordermode=-1);
   void           SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void   SetBorderMode(Short_t bordermode) {fBorderMode = bordermode;} // *MENU*
   virtual void   SetBorderSize(Short_t bordersize) {fBorderSize = bordersize;} // *MENU*

   ClassDefOverride(TWbox,1)  //A window box (box with 3-D effects)
};

#endif

