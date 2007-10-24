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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWbox                                                                //
//                                                                      //
// A window box (box with 3-D effects).                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TBox
#include "TBox.h"
#endif

#ifndef ROOT_TColor
#include "TColor.h"
#endif

class TWbox : public TBox {

protected:
   Short_t      fBorderSize;    //window box bordersize in pixels
   Short_t      fBorderMode;    //Bordermode (-1=down, 0 = no border, 1=up)

public:
   TWbox();
   TWbox(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
         Color_t color=18, Short_t bordersize=5 ,Short_t bordermode=1);
   TWbox(const TWbox &wbox);
   virtual ~TWbox();

   void          Copy(TObject &wbox) const;
   virtual void  Draw(Option_t *option="");
   virtual void  DrawWbox(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                      Color_t color=33 ,Short_t bordersize=5 ,Short_t bordermode=-1);
   virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
   Short_t       GetBorderMode() const { return fBorderMode;}
   Short_t       GetBorderSize() const { return fBorderSize;}
   Int_t         GetDarkColor() const  {return TColor::GetColorDark(GetFillColor());}
   Int_t         GetLightColor() const {return TColor::GetColorBright(GetFillColor());}
   virtual void  Paint(Option_t *option="");
   virtual void  PaintFrame(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                            Color_t color, Short_t bordersize, Short_t bordermode,
                            Bool_t tops);
   virtual void  PaintWbox(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2,
                 Color_t color=33, Short_t bordersize=5, Short_t bordermode=-1);
   virtual void  SavePrimitive(ostream &out, Option_t *option = "");
   virtual void  SetBorderMode(Short_t bordermode) {fBorderMode = bordermode;} // *MENU*
   virtual void  SetBorderSize(Short_t bordersize) {fBorderSize = bordersize;} // *MENU*

   ClassDef(TWbox,1)  //A window box (box with 3-D effects)
};

#endif

