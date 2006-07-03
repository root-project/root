// @(#)root/base:$Name:  $:$Id: TAttFill.cxx,v 1.12 2006/04/06 09:38:58 couet Exp $
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TAttFill.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"
#include "TVirtualPadEditor.h"
#include "TColor.h"

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
//  pattern_number can have any value from 1 to 25 (see table), or any value
// from 100 to 999. For the latest the numbering convention is the following:
//
//            pattern_number = ijk      (FillStyle = 3ijk)
//
//    i (1-9) : specify the space between each hatch
//              1 = 1/2mm  9 = 6mm
//
//    j (0-9) : specify angle between 0 and 90 degrees
//
//              0 = 0  
//              1 = 10
//              2 = 20
//              3 = 30
//              4 = 45
//              5 = Not drawn
//              6 = 60
//              7 = 70
//              8 = 80
//              9 = 90
//
//    k (0-9) : specify angle between 90 and 180 degrees
//              0 = 180
//              1 = 170
//              2 = 160
//              3 = 150
//              4 = 135
//              5 = Not drawn
//              6 = 120
//              7 = 110
//              8 = 100
//              9 = 90
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
void TAttFill::Copy(TAttFill &attfill) const
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
void TAttFill::SaveFillAttributes(ostream &out, const char *name, Int_t coldef, Int_t stydef)
{
    // Save fill attributes as C++ statement(s) on output stream out

   if (fFillColor != coldef) {
      if (fFillColor > 228) {
         TColor::SaveColor(out, fFillColor);
         out<<"   "<<name<<"->SetFillColor(ci);" << endl;
      } else 
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

   //if (gPad) gPad->UpdateFillAttributes(fFillColor,fFillStyle);
   
   TVirtualPadEditor::UpdateFillAttributes(fFillColor,fFillStyle);
}
