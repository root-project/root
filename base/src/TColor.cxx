// @(#)root/base:$Name:  $:$Id: TColor.cxx,v 1.1.1.1 2000/05/16 17:00:38 rdm Exp $
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream.h>

#include "TROOT.h"
#include "TColor.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"

ClassImp(TColor)

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*-* C O L O R class *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =================
//  At initialisation time, a table of colors is generated when
//  the first Canvas constructor is called.
//  This linked list can be accessed from the ROOT object
//      (see TROOT::GetListOfColors).
//  When a color is defined, two "companion" colors are also defined:
//    - the dark version (color_index + 100)
//    - the bright version (color_index + 150)
// The dark and bright color are used to give 3-D effects when drawing
// various boxes (see TWbox, TPave, TPaveText, TPaveLabel,etc).
//
//  This is the list of currently supported basic colors.
//  (here dark and bright colors are not shown).
//Begin_Html
/*
<img src="gif/colors.gif">
*/
//End_Html
//
//

//______________________________________________________________________________
TColor::TColor(): TNamed()
{
//*-*-*-*-*-*-*-*-*-*-*Color default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================

}

//______________________________________________________________________________
TColor::TColor(Int_t color, Float_t r, Float_t g, Float_t b, const char *name)
      :TNamed(name,"")
{
//*-*-*-*-*-*-*-*-*-*-*Color normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================
//-
//  Initialize a color structure.
//  Compute the RGB and HLS parameters

//*-*- Do not enter if color number already exist
   TColor *col = gROOT->GetColor(color);
   if (col) {
      Warning("TColor", "color %d already defined", color);
      return;
   }
   SetNumber(color);
   char aname[32];
   if (!strlen(name)) {
      sprintf(aname,"Color%d",color);
      SetName(aname);
   }
   const char *cname = GetName();
//*-*- enter in the list of colors
   gROOT->GetListOfColors()->AddAt(this,color);

   if (color > 0 && color < 51) {   //*-*- Now create associated colors for WBOX shading
      sprintf(aname,"%s%s",cname,"_dark");
      new TColor(100+color, 0,0,0,aname);
      sprintf(aname,"%s%s",cname,"_bright");
      new TColor(150+color, 0,0,0,aname);
   }

//*-*- Fill color structure
   SetRGB(r, g, b);
}

//______________________________________________________________________________
TColor::~TColor()
{
//*-*-*-*-*-*-*-*-*-*-*Color default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   gROOT->GetListOfColors()->Remove(this);
}

//______________________________________________________________________________
TColor::TColor(const TColor &color)
{
   ((TColor&)color).Copy(*this);
}

//______________________________________________________________________________
void TColor::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*-*-*-*Copy this color to color*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   TObject::Copy(obj);
   ((TColor&)obj).fRed   = fRed;
   ((TColor&)obj).fGreen = fGreen;
   ((TColor&)obj).fBlue  = fBlue;
   ((TColor&)obj).fHue   = fHue;
   ((TColor&)obj).fLight = fLight;
   ((TColor&)obj).fSaturation   = fSaturation;
}

//______________________________________________________________________________
void TColor::HLStoRGB(Float_t hue, Float_t light, Float_t satur, Float_t &r, Float_t &g, Float_t &b)
{
//*-*-*-*-*-*-*-*Compute HLS from RGB*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================
//- See HIGZ routine IGHTOR
//
   Float_t rh, rl, rs, rm1, rm2;
   rh = rl = rs = 0;
   if (hue   > 0) rh = hue;   if (rh > 360) rh = 360;
   if (light > 0) rl = light; if (rl > 1)   rl = 1;
   if (satur > 0) rs = satur; if (rs > 1)   rs = 1;

   if (rl <= 0.5) rm2 = rl*(1+rs);
   else           rm2 = rl + rs - rl*rs;
   rm1 = 2*rl - rm2;

   if (!rs) { r = rl; g = rl; b = rl; return;}
   r = HLStoRGB1(rm1, rm2, rh+120);
   g = HLStoRGB1(rm1, rm2, rh);
   b = HLStoRGB1(rm1, rm2, rh-120);
}

//______________________________________________________________________________
Float_t TColor::HLStoRGB1(Float_t rn1, Float_t rn2, Float_t huei)
{
//*-*-*-*-*-*-*-*Auxiliary to HLStoRGB*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            =====================
// See HIGZ routine IGHR01
//
   Float_t hue = huei;
   if (hue > 360) hue = hue - 360;
   if (hue < 0)   hue = hue + 360;
   if (hue < 60 ) return rn1 + (rn2-rn1)*hue/60;
   if (hue < 180) return rn2;
   if (hue < 240) return rn1 + (rn2-rn1)*(240-hue)/60;
   return rn1;
}

//______________________________________________________________________________
void TColor::ls(Option_t *) const
{
//*-*-*-*-*-*-*-*-*-*-*-*List this color with its attributes*-*-*-*-*-*-*-*-*
//*-*                    ===================================
   printf("Color:%d  Red=%f Green=%f Blue=%f\n",fNumber,fRed,fGreen,fBlue);
}

//______________________________________________________________________________
void TColor::Print(Option_t *) const
{
//*-*-*-*-*-*-*-*-*-*-*Dump this color with its attributes*-*-*-*-*-*-*-*-*-*
//*-*                  ===================================
   TColor::ls();
}

//______________________________________________________________________________
void TColor::RGBtoHLS(Float_t r, Float_t g, Float_t b, Float_t &hue, Float_t &light, Float_t &satur )
{
//*-*-*-*-*-*-*-*-*-*-*Compute HLS from RGB*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ====================
   Float_t rnorm, gnorm, bnorm, minval, maxval, msum, mdiff;
   minval = r;
   if (g <minval) minval = g;
   if (b <minval) minval = b;
   maxval = r;
   if (g >maxval) maxval = g;
   if (b >maxval) maxval = b;

   rnorm  = gnorm = bnorm = 0;
   mdiff  = maxval - minval;
   msum   = maxval + minval;
   light = 0.5*msum;
   if (maxval != minval) {
      rnorm = (maxval - r)/mdiff;
      gnorm = (maxval - g)/mdiff;
      bnorm = (maxval - b)/mdiff;
   }
   else { satur = hue = 0; return;}

   if (light <= 0.5) satur = mdiff/msum;
   else satur = mdiff/(2-msum);

   if (r == maxval) hue = 60.0 * (6.0 + bnorm - gnorm);
   if (g == maxval) hue = 60.0 * (2.0 + rnorm - bnorm);
   if (b == maxval) hue = 60.0 * (4.0 + gnorm - rnorm);
   if (hue > 360) hue = hue -360;
}

//______________________________________________________________________________
void TColor::SetRGB(Float_t r, Float_t g, Float_t b)
{
//*-*-*-*-*-*-*-*Initialize this color and its assosiated colors*-*-*-*-*-*-*
//*-*            ===============================================
   Int_t color = GetNumber();
   fRed   = r;
   fGreen = g;
   fBlue  = b;

   RGBtoHLS(r, g, b, fHue, fLight, fSaturation);

//*-*- make this color known to the graphics system
   if (gVirtualX && !gROOT->IsBatch())   gVirtualX->SetRGB(color, r, g, b);

   if (color > 50) return;
//*-*- Now define associated colors for WBOX shading
   Float_t dr, dg, db, lr, lg, lb;
   Int_t nplanes = 16;
   if (gVirtualX) gVirtualX->GetPlanes(nplanes);
//*-*-----------Dark color
//*-*           ==========
   HLStoRGB(fHue, 0.7*fLight, fSaturation, dr, dg, db);
   TColor *dark = gROOT->GetColor(100+color);
   if (dark) {
       if (nplanes > 8)  dark->SetRGB(dr, dg, db);
       else              dark->SetRGB(0.3,0.3,0.3);
    }
//*-*-----------Light color
//*-*           ===========
   HLStoRGB(fHue, 1.2*fLight, fSaturation, lr, lg, lb);
   TColor *light = gROOT->GetColor(150+color);
   if (light) {
      if (nplanes > 8) light->SetRGB(lr, lg, lb);
      else             light->SetRGB(0.8,0.8,0.8);
   }
}
