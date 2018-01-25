// @(#)root/hist:$Id$
// Author: Rene Brun   18/05/95

#include <string.h>
#include <stdio.h>
#include <ctype.h>

#include "TH1.h"


////////////////////////////////////////////////////////////////////////////////
/// Choose an axis according to "axis".

Int_t TH1::AxisChoice( Option_t *axis) const
{
   char achoice = toupper(axis[0]);
   if (achoice == 'X') return 1;
   if (achoice == 'Y') return 2;
   if (achoice == 'Z') return 3;
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the number of divisions for "axis".

Int_t TH1::GetNdivisions( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetNdivisions();
   if (ax == 2) return fYaxis.GetNdivisions();
   if (ax == 3) return fZaxis.GetNdivisions();
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the number of divisions for "axis".

Color_t TH1::GetAxisColor( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetAxisColor();
   if (ax == 2) return fYaxis.GetAxisColor();
   if (ax == 3) return fZaxis.GetAxisColor();
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the histogram bar offset along X or Y axis.
///
/// axis specifies which axis ("x","y"), default = "x"
///
///  - if axis="x"  The bar offset along the X axis is returned
///  - if axis="y"  The bar offset along the Y axis is returned

Float_t  TH1::GetBarOffset( Option_t *axis) const
{
   if (fBarOffset <= 1000) return Float_t(0.001*Float_t(fBarOffset));

   Int_t ax = AxisChoice(axis);
   if (ax == 1) {
      if (fBarOffset <= 3000 ) return Float_t(0.001*Float_t(fBarOffset-2000));
      if (fBarOffset <= 5000 ) return 0.;
   }
   if (ax == 2) {
      if (fBarOffset <= 3000 ) return 0.;
      if (fBarOffset <= 5000 ) return Float_t(0.001*Float_t(fBarOffset-4000));
   }

   return 0.;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the histogram bar width along X or Y axis.
///
/// axis specifies which axis ("x","y"), default = "x"
///
///  - if axis="x"  The bar width along the X axis is returned
///  - if axis="y"  The bar width along the Y axis is returned

Float_t TH1::GetBarWidth( Option_t *axis) const
{
   if (fBarWidth <= 1000) return Float_t(0.001*Float_t(fBarWidth));

   Int_t ax = AxisChoice(axis);
   if (ax == 1) {
      if (fBarWidth <= 3000 ) return Float_t(0.001*Float_t(fBarWidth-2000));
      if (fBarWidth <= 5000 ) return 1.;
   }
   if (ax == 2) {
      if (fBarWidth <= 3000 ) return 1.;
      if (fBarWidth <= 5000 ) return Float_t(0.001*Float_t(fBarWidth-4000));
   }

   return 1.;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the "axis" label color.

Color_t TH1::GetLabelColor( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelColor();
   if (ax == 2) return fYaxis.GetLabelColor();
   if (ax == 3) return fZaxis.GetLabelColor();
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the "axis" label font.

Style_t TH1::GetLabelFont( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelFont();
   if (ax == 2) return fYaxis.GetLabelFont();
   if (ax == 3) return fZaxis.GetLabelFont();
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the "axis" label offset.

Float_t TH1::GetLabelOffset( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelOffset();
   if (ax == 2) return fYaxis.GetLabelOffset();
   if (ax == 3) return fZaxis.GetLabelOffset();
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the "axis" label size.

Float_t TH1::GetLabelSize( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelSize();
   if (ax == 2) return fYaxis.GetLabelSize();
   if (ax == 3) return fZaxis.GetLabelSize();
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the "axis" tick length.

Float_t TH1::GetTickLength( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetTickLength();
   if (ax == 2) return fYaxis.GetTickLength();
   if (ax == 3) return fZaxis.GetTickLength();
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the "axis" title font.

Style_t TH1::GetTitleFont( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetTitleFont();
   if (ax == 2) return fYaxis.GetTitleFont();
   if (ax == 3) return fZaxis.GetTitleFont();
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the "axis" title offset.

Float_t TH1::GetTitleOffset( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetTitleOffset();
   if (ax == 2) return fYaxis.GetTitleOffset();
   if (ax == 3) return fZaxis.GetTitleOffset();
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the "axis" title size.

Float_t TH1::GetTitleSize( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetTitleSize();
   if (ax == 2) return fYaxis.GetTitleSize();
   if (ax == 3) return fZaxis.GetTitleSize();
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Set the number of divisions to draw an axis.
///
///  ndiv      : Number of divisions.
///
///       n = N1 + 100*N2 + 10000*N3
///       N1=number of primary divisions.
///       N2=number of secondary divisions.
///       N3=number of 3rd divisions.
///           e.g.:
///           nndi=0 --> no tick marks.
///           nndi=2 --> 2 divisions, one tick mark in the middle
///                      of the axis.
/// axis specifies which axis ("x","y","z"), default = "x"
/// if axis="xyz" set all 3 axes

void TH1::SetNdivisions(Int_t n, Option_t *axis)
{
   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetNdivisions(n);
   if (opt.Contains("y")) fYaxis.SetNdivisions(n);
   if (opt.Contains("z")) fZaxis.SetNdivisions(n);
}


////////////////////////////////////////////////////////////////////////////////
/// Set color to draw the axis line and tick marks.
///
/// axis specifies which axis ("x","y","z"), default = "x"
/// if axis="xyz" set all 3 axes

void TH1::SetAxisColor(Color_t color, Option_t *axis)
{
   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetAxisColor(color);
   if (opt.Contains("y")) fYaxis.SetAxisColor(color);
   if (opt.Contains("z")) fZaxis.SetAxisColor(color);
}


////////////////////////////////////////////////////////////////////////////////
/// Set the "axis" range.

void TH1::SetAxisRange(Axis_t xmin, Axis_t xmax, Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   TAxis *theAxis = 0;
   if (ax == 1) theAxis = GetXaxis();
   if (ax == 2) theAxis = GetYaxis();
   if (ax == 3) theAxis = GetZaxis();
   if (!theAxis) return;
   if (ax > fDimension) {
      SetMinimum(xmin);
      SetMaximum(xmax);
      return;
   }
   Int_t bin1 = theAxis->FindFixBin(xmin);
   Int_t bin2 = theAxis->FindFixBin(xmax);
   theAxis->SetRange(bin1, bin2);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the histogram bar offset along X and Y axis.
///
/// axis specifies which axis ("x","y"), default = "xy"
///
///  - if axis="xy" `offset` is set along X and Y axis
///  - if axis="x"  `offset` is set along the X axis and 0. along the Y axis
///  - if axis="y"  `offset` is set along the Y axis and 0. along the X axis
///
/// `offset` must be in the range [0.,1.]

void TH1::SetBarOffset(Float_t offset, Option_t *axis)
{
   if (offset < 0. || offset > 1.) {
      Error("SetBarOffset", "offset must be in the range [0.,1.]");
      return;
   }

   TString opt = axis;
   opt.ToLower();

   Bool_t barox = kFALSE; if (opt.Contains("x")) barox = kTRUE;
   Bool_t baroy = kFALSE; if (opt.Contains("y")) baroy = kTRUE;

   if (barox && baroy) {
      fBarOffset = Short_t(1000*offset);
      return;
   }
   if (barox) {
      fBarOffset = Short_t(1000*offset) + 2000;
      return;
   }
   if (baroy) {
      fBarOffset = Short_t(1000*offset) + 4000;
      return;
   }

   Warning("SetBarOffset","Invalid option");
}

////////////////////////////////////////////////////////////////////////////////
/// Set the histogram bar width along X and Y axis.
///
/// axis specifies which axis ("x","y"), default = "xy"
///
///  - if axis="xy" `width` is set along X and Y axis
///  - if axis="x"  `width` is set along the X axis and 1. along the Y axis
///  - if axis="y"  `width` is set along the Y axis and 1. along the X axis

void TH1::SetBarWidth(Float_t width, Option_t *axis)
{
   if (width < 0. || width > 1.) {
      Error("SetBarOffset", "width must be in the range [0.,1.]");
      return;
   }

   TString opt = axis;
   opt.ToLower();

   Bool_t barwx = kFALSE; if (opt.Contains("x")) barwx = kTRUE;
   Bool_t barwy = kFALSE; if (opt.Contains("y")) barwy = kTRUE;

   if (barwx && barwy) {
      fBarWidth = Short_t(1000*width);
      return;
   }
   if (barwx) {
      fBarWidth = Short_t(1000*width) + 2000;
      return;
   }
   if (barwy) {
      fBarWidth = Short_t(1000*width) + 4000;
      return;
   }

   Warning("SetBarWidth","Invalid option");
}

////////////////////////////////////////////////////////////////////////////////
/// Set axis labels color.
///
/// axis specifies which axis ("x","y","z"), default = "x"
/// if axis="xyz" set all 3 axes

void TH1::SetLabelColor(Color_t color, Option_t *axis)
{
   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetLabelColor(color);
   if (opt.Contains("y")) fYaxis.SetLabelColor(color);
   if (opt.Contains("z")) fZaxis.SetLabelColor(color);
}


////////////////////////////////////////////////////////////////////////////////
/// Set font number used to draw axis labels.
///
/// font  : Text font code = 10*fontnumber + precision
///         Font numbers must be between 1 and 14
///         precision = 1 fast hardware fonts (steps in the size)
///         precision = 2 scalable and rotatable hardware fonts
///
/// The default font number is 62.
/// axis specifies which axis ("x","y","z"), default = "x"
/// if axis="xyz" set all 3 axes

void TH1::SetLabelFont(Style_t font, Option_t *axis)
{
   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetLabelFont(font);
   if (opt.Contains("y")) fYaxis.SetLabelFont(font);
   if (opt.Contains("z")) fZaxis.SetLabelFont(font);
}


////////////////////////////////////////////////////////////////////////////////
/// Set offset between axis and axis' labels.
///
/// The offset is expressed as a percent of the pad height.
/// axis specifies which axis ("x","y","z"), default = "x"
/// if axis="xyz" set all 3 axes

void TH1::SetLabelOffset(Float_t offset, Option_t *axis)
{
   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetLabelOffset(offset);
   if (opt.Contains("y")) fYaxis.SetLabelOffset(offset);
   if (opt.Contains("z")) fZaxis.SetLabelOffset(offset);
}


////////////////////////////////////////////////////////////////////////////////
/// Set size of axis' labels.
///
/// The size is expressed as a percent of the pad height.
/// axis specifies which axis ("x","y","z"), default = "x"
/// if axis="xyz" set all 3 axes

void TH1::SetLabelSize(Float_t size, Option_t *axis)
{
   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetLabelSize(size);
   if (opt.Contains("y")) fYaxis.SetLabelSize(size);
   if (opt.Contains("z")) fZaxis.SetLabelSize(size);
}


////////////////////////////////////////////////////////////////////////////////
/// Set the axis' tick marks length.
///
/// axis specifies which axis ("x","y","z"), default = "x"
/// if axis="xyz" set all 3 axes

void TH1::SetTickLength(Float_t length, Option_t *axis)
{
   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetTickLength(length);
   if (opt.Contains("y")) fYaxis.SetTickLength(length);
   if (opt.Contains("z")) fZaxis.SetTickLength(length);
}


////////////////////////////////////////////////////////////////////////////////
/// Set the axis' title font.
///
///  - if axis =="x"  set the X axis title font
///  - if axis =="y"  set the Y axis title font
///  - if axis =="z"  set the Z axis title font
/// any other value of axis will set the pad title font
///
/// if axis="xyz" set all 3 axes

void TH1::SetTitleFont(Style_t font, Option_t *axis)
{
   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetTitleFont(font);
   if (opt.Contains("y")) fYaxis.SetTitleFont(font);
   if (opt.Contains("z")) fZaxis.SetTitleFont(font);
}


////////////////////////////////////////////////////////////////////////////////
/// Specify a parameter offset to control the distance between the axis
/// and the axis' title.
///
///  - offset = 1 means : use the default distance
///  - offset = 1.2 means: the distance will be 1.2*(default distance)
///  - offset = 0.8 means: the distance will be 0.8*(default distance)
///
/// axis specifies which axis ("x","y","z"), default = "x"
/// if axis="xyz" set all 3 axes

void TH1::SetTitleOffset(Float_t offset, Option_t *axis)
{
   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetTitleOffset(offset);
   if (opt.Contains("y")) fYaxis.SetTitleOffset(offset);
   if (opt.Contains("z")) fZaxis.SetTitleOffset(offset);
}


////////////////////////////////////////////////////////////////////////////////
/// Set the axis' title size.
///
///  - if axis = "x" set the X axis title size
///  - if axis = "y" set the Y axis title size
///  - if axis = "z" set the Z axis title size
///
/// if axis ="xyz" set all 3 axes

void TH1::SetTitleSize(Float_t size, Option_t *axis)
{
   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetTitleSize(size);
   if (opt.Contains("y")) fYaxis.SetTitleSize(size);
   if (opt.Contains("z")) fZaxis.SetTitleSize(size);
}
