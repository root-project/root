// @(#)root/postscript:$Name:  $:$Id: TImageDump.h,v 1.4 2005/05/15 05:53:45 brun Exp $
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
class TColor;
class TImageDump : public TVirtualPS {
protected:
   TImage  *fImage;     // image
	Int_t    fType;      // PostScript workstation type

   void Add2PixelBuffer(Short_t x, Short_t y, TColor *c);
   void Add2HLinesBuffer(Short_t x, Short_t y1, Short_t y2, TColor *c);
   void Add2VLinesBuffer(Short_t y, Short_t x1, Short_t x2, TColor *c);
   void FlushBuffers();

public:
   TImageDump();
   TImageDump(const char *filename, Int_t type = -111);
   virtual ~TImageDump();

   void  CellArrayBegin(Int_t W, Int_t H, Double_t x1, Double_t x2, Double_t y1, Double_t y2);
   void  CellArrayFill(Int_t r, Int_t g, Int_t b);
   void  CellArrayEnd();
   void  Close(Option_t *opt = "");
   void  DrawBox(Double_t x1, Double_t y1,Double_t x2, Double_t  y2);
   void  DrawFrame(Double_t xl, Double_t yl, Double_t xt, Double_t  yt,
                   Int_t mode, Int_t border, Int_t dark, Int_t light);
   void  DrawPolyMarker(Int_t n, Float_t *x, Float_t *y);
   void  DrawPolyMarker(Int_t n, Double_t *x, Double_t *y);
   void  DrawPS(Int_t n, Float_t *xw, Float_t *yw);
   void  DrawPS(Int_t n, Double_t *xw, Double_t *yw);
   void  NewPage();
   void  Open(const char *filename, Int_t type = -111);
   void  Text(Double_t x, Double_t y, const char *string);
   void  SetColor(Float_t r, Float_t g, Float_t b);
   void *GetStream() const {  return (void*)fImage; }
   void  SetType(Int_t type = -111) { fType = type; }
   Int_t GetType() const { return fType; }

   ClassDef(TImageDump,0)  // create image in batch mode
};

#endif
