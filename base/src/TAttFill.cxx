// @(#)root/base:$Name:  $:$Id: TAttFill.cxx,v 1.1.1.1 2000/05/16 17:00:38 rdm Exp $
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <fstream.h>

#include "TROOT.h"
#include "TAttFill.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"

ClassImp(TAttFill)

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*-*Fill Area Attributes class*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==========================
//*-*  Fill Area attributes are:
//*-*    Fill Color
//*-*    Fill Style
//*-*
//*-*  This class is used (in general by secondary inheritance)
//*-*  by many other classes (graphics, histograms).
//*-*
//*-*  The following table shows the list of default colors.
//Begin_Html
/*
<img src="gif/colors.gif">
*/
//End_Html
//
//  Conventions for fill styles:
//    0    : hollow
//    1001 : Solid
//    2001 : hatch style
//    3000+pattern_number (see below)
//    4000 :the window is transparent.
//    4000 to 4100 the window is 100% transparent to 100% opaque
//
//*-*  The following table shows the list of pattern styles.
//Begin_Html
/*
<img src="gif/fillstyles.gif">
*/
//End_Html
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//______________________________________________________________________________
TAttFill::TAttFill()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttFill default constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===========================
//*-*  Default fill attributes are taking from the current style
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   if (!gStyle) return;
   fFillColor = gStyle->GetFillColor();
   fFillStyle = gStyle->GetFillStyle();
}

//______________________________________________________________________________
TAttFill::TAttFill(Color_t color, Style_t style)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttFill normal constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===========================
//*-*    color Fill Color
//*-*    style Fill Style
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   fFillColor = color;
   fFillStyle = style;
}

//______________________________________________________________________________
TAttFill::~TAttFill()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttFill destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =================
}

//______________________________________________________________________________
void TAttFill::Copy(TAttFill &attfill)
{
//*-*-*-*-*-*-*-*-*Copy this fill attributes to a new attfill*-*-*-*-*-*-*-*-*
//*-*              ==========================================
   attfill.fFillColor  = fFillColor;
   attfill.fFillStyle  = fFillStyle;
}

//______________________________________________________________________________
void TAttFill::Modify()
{
//*-*-*-*-*-*-*-*Change current fill area attributes if necessary*-*-*-*-*-*-*
//*-*            ================================================

   if (!gPad) return;
   if (!gPad->IsBatch()) {
      gVirtualX->SetFillColor(fFillColor);
      gVirtualX->SetFillStyle(fFillStyle);
   }

   gPad->SetAttFillPS(fFillColor,fFillStyle);
}

//______________________________________________________________________________
void TAttFill::ResetAttFill(Option_t *)
{
//*-*-*-*-*-*-*-*-*Reset this fill attributes to default values*-*-*-*-*-*-*
//*-*              ============================================
  fFillColor = 1;
  fFillStyle = 0;
}

//______________________________________________________________________________
void TAttFill::SaveFillAttributes(ofstream &out, const char *name, Int_t coldef, Int_t stydef)
{
    // Save fill attributes as C++ statement(s) on output stream out

   if (fFillColor != coldef) {
      out<<"   "<<name<<"->SetFillColor("<<fFillColor<<");"<<endl;
   }
   if (fFillStyle != stydef) {
      out<<"   "<<name<<"->SetFillStyle("<<fFillStyle<<");"<<endl;
   }
}

//______________________________________________________________________________
void TAttFill::SetFillAttributes()
{
//*-*-*-*-*-*-*-*-*Invoke the DialogCanvas Fill attributes*-*-*-*-*-*-*
//*-*              =======================================

   if (gPad) gROOT->SetSelectedPad(gPad->GetSelectedPad());

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   if (!lc->FindObject("R__attfill")) {
      gROOT->ProcessLine("TAttFillCanvas *R__attfill = "
                         "new TAttFillCanvas(\"R__attfill\",\"Fill Attributes\","
                         "250,400);");
   }
   gROOT->ProcessLine(Form("R__attfill->UpdateFillAttributes(%d,%d);"
                           "R__attfill->Show();",fFillColor,fFillStyle));
}
