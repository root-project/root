// @(#)root/base:$Name$:$Id$
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
        Coord_t     fLeftMargin;      //LeftMargin
        Coord_t     fRightMargin;     //RightMargin
        Coord_t     fBottomMargin;    //BottomMargin
        Coord_t     fTopMargin;       //TopMargin
        Coord_t     fXfile;           //X position where to draw the file name
        Coord_t     fYfile;           //X position where to draw the file name
        Coord_t     fAfile;           //Alignment for the file name
        Coord_t     fXstat;           //X position where to draw the statistics
        Coord_t     fYstat;           //X position where to draw the statistics
        Coord_t     fAstat;           //Alignment for the statistics
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
        Coord_t          GetBottomMargin() { return fBottomMargin;}
        Coord_t          GetLeftMargin() { return fLeftMargin;}
        Coord_t          GetRightMargin() { return fRightMargin;}
        Coord_t          GetTopMargin() { return fTopMargin;}
        Coord_t          GetAfile() { return fAfile;}
        Coord_t          GetXfile() { return fXfile;}
        Coord_t          GetYfile() { return fYfile;}
        Coord_t          GetAstat() { return fAstat;}
        Coord_t          GetXstat() { return fXstat;}
        Coord_t          GetYstat() { return fYstat;}
        Color_t          GetFrameFillColor() {return fFrameFillColor;}
        Color_t          GetFrameLineColor() {return fFrameLineColor;}
        Style_t          GetFrameFillStyle() {return fFrameFillStyle;}
        Style_t          GetFrameLineStyle() {return fFrameLineStyle;}
        Width_t          GetFrameLineWidth() {return fFrameLineWidth;}
        Width_t          GetFrameBorderSize() {return fFrameBorderSize;}
        Int_t            GetFrameBorderMode() {return fFrameBorderMode;}
        virtual void     Print(Option_t *option="");
        virtual void     ResetAttPad(Option_t *option="");
        virtual void     SetBottomMargin(Coord_t bottommargin);
        virtual void     SetLeftMargin(Coord_t leftmargin);
        virtual void     SetRightMargin(Coord_t rightmargin);
        virtual void     SetTopMargin(Coord_t topmargin);
        virtual void     SetAfile(Coord_t afile) { fAfile=afile;}
        virtual void     SetXfile(Coord_t xfile) { fXfile=xfile;}
        virtual void     SetYfile(Coord_t yfile) { fYfile=yfile;}
        virtual void     SetAstat(Coord_t astat) { fAstat=astat;}
        virtual void     SetXstat(Coord_t xstat) { fXstat=xstat;}
        virtual void     SetYstat(Coord_t ystat) { fYstat=ystat;}
        void             SetFrameFillColor(Color_t color=1) {fFrameFillColor = color;}
        void             SetFrameLineColor(Color_t color=1) {fFrameLineColor = color;}
        void             SetFrameFillStyle(Style_t styl=0)  {fFrameFillStyle = styl;}
        void             SetFrameLineStyle(Style_t styl=0)  {fFrameLineStyle = styl;}
        void             SetFrameLineWidth(Width_t width=1) {fFrameLineWidth = width;}
        void             SetFrameBorderSize(Width_t size=1) {fFrameBorderSize = size;}
        void             SetFrameBorderMode(Int_t mode=1) {fFrameBorderMode = mode;}

        ClassDef(TAttPad,2)  //Pad attributes
};

#endif

