// @(#)root/base:$Name$:$Id$
// Author: Rene Brun   12/05/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <fstream.h>

#include "TROOT.h"
#include "Strlen.h"
#include "TAttMarker.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"

ClassImp(TAttMarker)

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*-*Marker Attributes class*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =======================
//*-*  Marker attributes are:
//*-*    Marker Color
//*-*    Marker style
//*-*    Marker Size
//*-*
//*-*  This class is used (in general by secondary inheritance)
//*-*  by many other classes (graphics, histograms).
//*-*
//*-*  List of the currently supported markers (screen and PostScript)
//*-*  ===============================================================
//*-*      1 : dot
//*-*      2 : +
//*-*      3 : *
//*-*      4 : o (open circle)
//*-*      5 : x
//*-*      6 : full circle (small)
//*-*      7 : full circle (medium)
//*-*      8 : full circle (large)
//*-*      9 -->19 : dot
//*-*     20 : full circle
//*-*     21 : full square
//*-*     22 : full triangle up
//*-*     23 : full triangle down
//*-*     24 : open circle
//*-*     25 : open square
//*-*     26 : open triangle up
//*-*     27 : open diamond
//*-*     28 : open cross
//*-*     29 : open star
//*-*     30 : full star
//*-*
//Begin_Html
/*
<img src="gif/markers.gif">
*/
//End_Html
//*-*
//*-*   Various marker sizes are shown in the figure below.
//*-*   The default marker size=1 is shown in the botton left corner.
//*-*   Marker sizes smaller than 1 can be specified.
//*-*
//Begin_Html
/*
<img src="gif/markersize.gif">
*/
//End_Html
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//______________________________________________________________________________
TAttMarker::TAttMarker()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttMarker default constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================
//*-*  Default text attributes are taking from the current style
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   if (!gStyle) return;
   fMarkerColor = gStyle->GetMarkerColor();
   fMarkerStyle = gStyle->GetMarkerStyle();
   fMarkerSize  = gStyle->GetMarkerSize();
}

//______________________________________________________________________________
TAttMarker::TAttMarker(Color_t color, Style_t style, Size_t msize)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttMarker normal constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ============================
//*-*  Text attributes are taking from the argument list
//*-*    color : Marker Color Index
//*-*    style : Marker style (from 1 to 30)
//*-*    size  : marker size (float)
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   fMarkerColor = color;
   fMarkerSize  = msize;
   fMarkerStyle = style;
}

//______________________________________________________________________________
TAttMarker::~TAttMarker()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttMarker destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ====================

}

//______________________________________________________________________________
void TAttMarker::Copy(TAttMarker &attmarker)
{
//*-*-*-*-*-*-*-*-*Copy this marker attributes to a new atttext*-*-*-*-*-*-*-*
//*-*              ============================================
   attmarker.fMarkerColor  = fMarkerColor;
   attmarker.fMarkerStyle  = fMarkerStyle;
   attmarker.fMarkerSize   = fMarkerSize;
}

//______________________________________________________________________________
void TAttMarker::Modify()
{
//*-*-*-*-*-*-*-*-*-*Change current marker attributes if necessary*-*-*-*-*-*-*
//*-*                =============================================

   if (!gPad) return;
   if (!gPad->IsBatch()) {
      gVirtualX->SetMarkerColor(fMarkerColor);
      gVirtualX->SetMarkerSize (fMarkerSize);
      gVirtualX->SetMarkerStyle(fMarkerStyle);
   }

   gPad->SetAttMarkerPS(fMarkerColor,fMarkerStyle,fMarkerSize);
}

//______________________________________________________________________________
void TAttMarker::ResetAttMarker(Option_t *)
{
//*-*-*-*-*-*-*-*-*Reset this marker attributes to default values*-*-*-*-*-*-*
//*-*              ==============================================

   fMarkerColor  = 1;
   fMarkerStyle  = 1;
   fMarkerSize   = 1;
}

//______________________________________________________________________________
void TAttMarker::SaveMarkerAttributes(ofstream &out, const char *name, Int_t coldef, Int_t stydef, Int_t sizdef)
{
    // Save line attributes as C++ statement(s) on output stream out

   if (fMarkerColor != coldef) {
      out<<"   "<<name<<"->SetMarkerColor("<<fMarkerColor<<");"<<endl;
   }
   if (fMarkerStyle != stydef) {
      out<<"   "<<name<<"->SetMarkerStyle("<<fMarkerStyle<<");"<<endl;
   }
   if (fMarkerSize != sizdef) {
      out<<"   "<<name<<"->SetMarkerSize("<<fMarkerSize<<");"<<endl;
   }

}

//______________________________________________________________________________
void TAttMarker::SetMarkerAttributes()
{
//*-*-*-*-*-*-*-*-*Invoke the DialogCanvas Marker attributes*-*-*-*-*-*-*
//*-*              =========================================

   if (gPad) gROOT->SetSelectedPad(gPad->GetSelectedPad());

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   if (!lc->FindObject("R__attmarker")) {
      gROOT->ProcessLine("TAttMarkerCanvas *R__attmarker = "
                         "new TAttMarkerCanvas(\"R__attmarker\",\"Marker Attributes\","
                         "250,400);");
   }
   gROOT->ProcessLine(Form("R__attmarker->UpdateMarkerAttributes(%d,%d,%f);"
                           "R__attmarker->Show();",fMarkerColor,fMarkerStyle,fMarkerSize));
}
