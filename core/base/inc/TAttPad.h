// @(#)root/base:$Id$
// Author: Rene Brun   04/01/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttPad
#define ROOT_TAttPad


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttPad                                                              //
//                                                                      //
// Pad attributes.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif


class TAttPad {
protected:
   Float_t     fLeftMargin;      //LeftMargin
   Float_t     fRightMargin;     //RightMargin
   Float_t     fBottomMargin;    //BottomMargin
   Float_t     fTopMargin;       //TopMargin
   Float_t     fXfile;           //X position where to draw the file name
   Float_t     fYfile;           //Y position where to draw the file name
   Float_t     fAfile;           //Alignment for the file name
   Float_t     fXstat;           //X position where to draw the statistics
   Float_t     fYstat;           //Y position where to draw the statistics
   Float_t     fAstat;           //Alignment for the statistics
   Color_t     fFrameFillColor;  //pad frame fill color
   Color_t     fFrameLineColor;  //pad frame line color
   Style_t     fFrameFillStyle;  //pad frame fill style
   Style_t     fFrameLineStyle;  //pad frame line style
   Width_t     fFrameLineWidth;  //pad frame line width
   Width_t     fFrameBorderSize; //pad frame border size
   Int_t       fFrameBorderMode; //pad frame border mode

public:
   TAttPad();
   virtual ~TAttPad();
   virtual void     Copy(TAttPad &attpad) const;
   Float_t          GetBottomMargin() const { return fBottomMargin;}
   Float_t          GetLeftMargin() const { return fLeftMargin;}
   Float_t          GetRightMargin() const { return fRightMargin;}
   Float_t          GetTopMargin() const { return fTopMargin;}
   Float_t          GetAfile() const { return fAfile;}
   Float_t          GetXfile() const { return fXfile;}
   Float_t          GetYfile() const { return fYfile;}
   Float_t          GetAstat() const { return fAstat;}
   Float_t          GetXstat() const { return fXstat;}
   Float_t          GetYstat() const { return fYstat;}
   Color_t          GetFrameFillColor() const {return fFrameFillColor;}
   Color_t          GetFrameLineColor() const {return fFrameLineColor;}
   Style_t          GetFrameFillStyle() const {return fFrameFillStyle;}
   Style_t          GetFrameLineStyle() const {return fFrameLineStyle;}
   Width_t          GetFrameLineWidth() const {return fFrameLineWidth;}
   Width_t          GetFrameBorderSize() const {return fFrameBorderSize;}
   Int_t            GetFrameBorderMode() const {return fFrameBorderMode;}
   virtual void     Print(Option_t *option="") const;
   virtual void     ResetAttPad(Option_t *option="");
   virtual void     SetBottomMargin(Float_t bottommargin);
   virtual void     SetLeftMargin(Float_t leftmargin);
   virtual void     SetRightMargin(Float_t rightmargin);
   virtual void     SetTopMargin(Float_t topmargin);
   virtual void     SetMargin(Float_t left, Float_t right, Float_t bottom, Float_t top);
   virtual void     SetAfile(Float_t afile) { fAfile=afile;}
   virtual void     SetXfile(Float_t xfile) { fXfile=xfile;}
   virtual void     SetYfile(Float_t yfile) { fYfile=yfile;}
   virtual void     SetAstat(Float_t astat) { fAstat=astat;}
   virtual void     SetXstat(Float_t xstat) { fXstat=xstat;}
   virtual void     SetYstat(Float_t ystat) { fYstat=ystat;}
   void             SetFrameFillColor(Color_t color=1) {fFrameFillColor = color;}
   void             SetFrameLineColor(Color_t color=1) {fFrameLineColor = color;}
   void             SetFrameFillStyle(Style_t styl=0)  {fFrameFillStyle = styl;}
   void             SetFrameLineStyle(Style_t styl=0)  {fFrameLineStyle = styl;}
   void             SetFrameLineWidth(Width_t width=1) {fFrameLineWidth = width;}
   void             SetFrameBorderSize(Width_t size=1) {fFrameBorderSize = size;}
   void             SetFrameBorderMode(Int_t mode=1) {fFrameBorderMode = mode;}

   ClassDef(TAttPad,3);  //Pad attributes
};

#endif

