// @(#)root/hist:$Id$
// Author: Rene Brun   05/04/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TF12.h"
#include "TH1.h"
#include "TVirtualPad.h"

ClassImp(TF12);

/** \class TF12
    \ingroup Functions
 A projection of a TF2 along X or Y

It has the same behaviour as a TF1

Example of a function

~~~ {.cpp}
    TF2 *f2 = new TF2("f2","sin(x)*sin(y)/(x*y)",0,5,0,5);
    TF12 *f12 = new TF12("f12",f2,0.1,"y");
    f12->Draw();
~~~

*/

////////////////////////////////////////////////////////////////////////////////
/// TF12 default constructor

TF12::TF12(): TF1()
{
   fCase = 0;
   fF2   = 0;
   fXY   = 0;
}


////////////////////////////////////////////////////////////////////////////////
/// TF12 normal constructor.
///
/// Create a TF12 (special TF1) from a projection of a TF2
/// for a fix value of Y if option="X" or X if option="Y"
/// This value may be changed at any time via TF12::SetXY(xy)

TF12::TF12(const char *name, TF2 *f2, Double_t xy, Option_t *option)
      :TF1(name,"x",0,0)
{
   SetName(name);
   fF2 = f2;
   TString opt=option;
   opt.ToLower();
   if (!f2) {
      Error("TF12","Pointer to TF2 is null");
      return;
   }
   SetXY(xy);
   if (opt.Contains("y")) {
      fXmin = f2->GetYmin();
      fXmax = f2->GetYmax();
      fCase = 1;
   } else {
      fXmin = f2->GetXmin();
      fXmax = f2->GetXmax();
      fCase = 0;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// F2 default destructor.

TF12::~TF12()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TF12::TF12(const TF12 &f12) : TF1(f12)
{
   ((TF12&)f12).Copy(*this);
}


////////////////////////////////////////////////////////////////////////////////
/// Copy this F2 to a new F2.

void TF12::Copy(TObject &obj) const
{
   TF1::Copy(obj);
   ((TF12&)obj).fXY      = fXY;
   ((TF12&)obj).fCase    = fCase;
   ((TF12&)obj).fF2      = fF2;
}


////////////////////////////////////////////////////////////////////////////////
/// Draw a copy of this function with its current attributes.
///
/// This function MUST be used instead of Draw when you want to draw
/// the same function with different parameters settings in the same canvas.
///
/// Possible option values are:
///
/// option | description
/// -------|----------------------------------------
/// "SAME" | superimpose on top of existing picture
/// "L"    | connect all computed points with a straight line
/// "C"    | connect all computed points with a smooth curve
///
/// Note that the default value is "F". Therefore to draw on top
/// of an existing picture, specify option "SL"

TF1 *TF12::DrawCopy(Option_t *option) const
{
   TF12 *newf2 = new TF12();
   Copy(*newf2);
   newf2->AppendPad(option);
   newf2->SetBit(kCanDelete);
   return newf2;
}


////////////////////////////////////////////////////////////////////////////////
/// Evaluate this formula
///
///   Computes the value of the referenced TF2 for a fix value of X or Y

Double_t TF12::Eval(Double_t x, Double_t /*y*/, Double_t /*z*/, Double_t /*t*/) const
{
   if (!fF2) return 0;
   if (fCase == 0) {
      return fF2->Eval(x,fXY,0);
   } else {
      return fF2->Eval(fXY,x,0);
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Evaluate this function at point x[0]
///
/// x[0] is the value along X if fCase =0, the value along Y if fCase=1
/// if params is non null, the array will be used instead of the internal TF2
/// parameters

Double_t TF12::EvalPar(const Double_t *x, const Double_t *params)
{
   if (!fF2) return 0;
   Double_t xx[2];
   if (fCase == 0) {
      xx[0] = x[0];
      xx[1] = fXY;
   } else {
      xx[0] = fXY;
      xx[1] = x[0];
   }
   fF2->InitArgs(xx,params);
   return fF2->EvalPar(xx,params);
}


////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TF12::SavePrimitive(std::ostream & /*out*/, Option_t * /*option*/ /*= ""*/)
{
   Error("SavePrimitive","Function not yet implemented");
}


////////////////////////////////////////////////////////////////////////////////
/// Set the value of the constant for the TF2
///
///   constant in X when projecting along Y
///   constant in Y when projecting along X
///  The function title is set to include the value of the constant
///  The current pad is updated

void TF12::SetXY(Double_t xy)
{
   fXY = xy;
   if (!fF2) return;
   if (fCase == 0) SetTitle(Form("%s (y=%g)",fF2->GetTitle(),xy));
   else            SetTitle(Form("%s (x=%g)",fF2->GetTitle(),xy));
   if (fHistogram) fHistogram->SetTitle(GetTitle());
   if (gPad) gPad->Modified();
}
