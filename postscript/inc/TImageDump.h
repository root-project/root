// @(#)root/postscript:$Name:  $:$Id: TImageDump.h,v 1.2 2005/05/03 13:11:32 brun Exp $
// Author: Valeriy Onuchin   29/04/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TImageDump
#define ROOT_TImageDump


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TImageDump                                                           //
//                                                                      //
// save canvas in an image in batch mode                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualPS
#include "TVirtualPS.h"
#endif

class TImage;
class TImageDump : public TVirtualPS {
protected:
   TImage  *fImage;     // image
	Int_t    fType;      // PostScript workstation type
  
public:
   TImageDump();
   TImageDump(const char *filename, Int_t type=-111);
   virtual ~TImageDump();

   virtual void  CellArrayBegin(Int_t W, Int_t H, Double_t x1, Double_t x2, Double_t y1, Double_t y2);
   virtual void  CellArrayFill(Int_t r, Int_t g, Int_t b);
   virtual void  CellArrayEnd();
   virtual void  Close(Option_t *opt="");
   virtual void  DrawBox(Double_t x1, Double_t y1,Double_t x2, Double_t  y2);
   virtual void  DrawFrame(Double_t xl, Double_t yl, Double_t xt, Double_t  yt,
                           Int_t mode, Int_t border, Int_t dark, Int_t light);
   virtual void  DrawPolyMarker(Int_t n, Float_t *x, Float_t *y);
   virtual void  DrawPolyMarker(Int_t n, Double_t *x, Double_t *y);
   virtual void  DrawPS(Int_t n, Float_t *xw, Float_t *yw);
   virtual void  DrawPS(Int_t n, Double_t *xw, Double_t *yw);
   virtual void  NewPage();
   virtual void  Open(const char *filename, Int_t type=-111);
   virtual void  Text(Double_t x, Double_t y, const char *string);
   virtual void  SetColor(Float_t r, Float_t g, Float_t b);
   virtual void *GetStream() const {  return (void*)fImage; }
   virtual void  SetType(Int_t type=-111) { fType = type; }
   virtual Int_t GetType() const { return fType; }

   ClassDef(TImageDump,0)  // create image in batch mode
};

#endif
