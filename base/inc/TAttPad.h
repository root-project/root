// @(#)root/base:$Name:  $:$Id: TAttPad.h,v 1.2 2000/06/13 12:35:10 brun Exp $
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

#ifndef ROOT_Gtypes
#include "Gtypes.h"
#endif

#ifndef ROOT_Htypes
#include "Htypes.h"
#endif


class TAttPad {
protected:
        Float_t     fLeftMargin;      //LeftMargin
        Float_t     fRightMargin;     //RightMargin
        Float_t     fBottomMargin;    //BottomMargin
        Float_t     fTopMargin;       //TopMargin
        Float_t     fXfile;           //X position where to draw the file name
        Float_t     fYfile;           //X position where to draw the file name
        Float_t     fAfile;           //Alignment for the file name
        Float_t     fXstat;           //X position where to draw the statistics
        Float_t     fYstat;           //X position where to draw the statistics
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
        virtual void     Copy(TAttPad &attpad);
        Float_t          GetBottomMargin() { return fBottomMargin;}
        Float_t          GetLeftMargin() { return fLeftMargin;}
        Float_t          GetRightMargin() { return fRightMargin;}
        Float_t          GetTopMargin() { return fTopMargin;}
        Float_t          GetAfile() { return fAfile;}
        Float_t          GetXfile() { return fXfile;}
        Float_t          GetYfile() { return fYfile;}
        Float_t          GetAstat() { return fAstat;}
        Float_t          GetXstat() { return fXstat;}
        Float_t          GetYstat() { return fYstat;}
        Color_t          GetFrameFillColor() {return fFrameFillColor;}
        Color_t          GetFrameLineColor() {return fFrameLineColor;}
        Style_t          GetFrameFillStyle() {return fFrameFillStyle;}
        Style_t          GetFrameLineStyle() {return fFrameLineStyle;}
        Width_t          GetFrameLineWidth() {return fFrameLineWidth;}
        Width_t          GetFrameBorderSize() {return fFrameBorderSize;}
        Int_t            GetFrameBorderMode() {return fFrameBorderMode;}
        virtual void     Print(Option_t *option="");
        virtual void     ResetAttPad(Option_t *option="");
        virtual void     SetBottomMargin(Float_t bottommargin);
        virtual void     SetLeftMargin(Float_t leftmargin);
        virtual void     SetRightMargin(Float_t rightmargin);
        virtual void     SetTopMargin(Float_t topmargin);
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

        ClassDef(TAttPad,3)  //Pad attributes
};

#endif

