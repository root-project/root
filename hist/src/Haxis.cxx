// @(#)root/hist:$Name:  $:$Id: Haxis.cxx,v 1.1.1.1 2000/05/16 17:00:41 rdm Exp $
// Author: Rene Brun   18/05/95
// ---------------------------------- haxis.C

#include <string.h>
#include <stdio.h>
#include <ctype.h>

#include "TH1.h"


//______________________________________________________________________________
Int_t TH1::AxisChoice( Option_t *axis)
{
   char achoice = toupper(axis[0]);
   if (achoice == 'X') return 1;
   if (achoice == 'Y') return 2;
   if (achoice == 'Z') return 3;
   return 0;
}

//______________________________________________________________________________
Int_t TH1::GetNdivisions( Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetNdivisions();
   if (ax == 2) return fYaxis.GetNdivisions();
   if (ax == 3) return fZaxis.GetNdivisions();
   return 0;
}

//______________________________________________________________________________
Color_t TH1::GetAxisColor( Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetAxisColor();
   if (ax == 2) return fYaxis.GetAxisColor();
   if (ax == 3) return fZaxis.GetAxisColor();
   return 0;
}

//______________________________________________________________________________
Color_t TH1::GetLabelColor( Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelColor();
   if (ax == 2) return fYaxis.GetLabelColor();
   if (ax == 3) return fZaxis.GetLabelColor();
   return 0;
}

//______________________________________________________________________________
Style_t TH1::GetLabelFont( Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelFont();
   if (ax == 2) return fYaxis.GetLabelFont();
   if (ax == 3) return fZaxis.GetLabelFont();
   return 0;
}

//______________________________________________________________________________
Float_t TH1::GetLabelOffset( Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelOffset();
   if (ax == 2) return fYaxis.GetLabelOffset();
   if (ax == 3) return fZaxis.GetLabelOffset();
   return 0;
}

//______________________________________________________________________________
Float_t TH1::GetLabelSize( Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelSize();
   if (ax == 2) return fYaxis.GetLabelSize();
   if (ax == 3) return fZaxis.GetLabelSize();
   return 0;
}

//______________________________________________________________________________
Float_t TH1::GetTickLength( Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetTickLength();
   if (ax == 2) return fYaxis.GetTickLength();
   if (ax == 3) return fZaxis.GetTickLength();
   return 0;
}

//______________________________________________________________________________
Float_t TH1::GetTitleOffset( Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetTitleOffset();
   if (ax == 2) return fYaxis.GetTitleOffset();
   if (ax == 3) return fZaxis.GetTitleOffset();
   return 0;
}

//______________________________________________________________________________
Float_t TH1::GetTitleSize( Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetTitleSize();
   if (ax == 2) return fYaxis.GetTitleSize();
   if (ax == 3) return fZaxis.GetTitleSize();
   return 0;
}

//______________________________________________________________________________
void TH1::SetNdivisions(Int_t n, Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) fXaxis.SetNdivisions(n);
   if (ax == 2) fYaxis.SetNdivisions(n);
   if (ax == 3) fZaxis.SetNdivisions(n);
}

//______________________________________________________________________________
void TH1::SetAxisColor(Color_t color, Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) fXaxis.SetAxisColor(color);
   if (ax == 2) fYaxis.SetAxisColor(color);
   if (ax == 3) fZaxis.SetAxisColor(color);
}

//______________________________________________________________________________
void TH1::SetAxisRange(Axis_t xmin, Axis_t xmax, Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   TAxis *theAxis = 0;
   if (ax == 1) theAxis = GetXaxis();
   if (ax == 2) theAxis = GetYaxis();
   if (ax == 3) theAxis = GetZaxis();
   Int_t bin1 = theAxis->FindFixBin(xmin);
   Int_t bin2 = theAxis->FindFixBin(xmax);
   theAxis->SetRange(bin1, bin2);
}

//______________________________________________________________________________
void TH1::SetLabelColor(Color_t color, Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) fXaxis.SetLabelColor(color);
   if (ax == 2) fYaxis.SetLabelColor(color);
   if (ax == 3) fZaxis.SetLabelColor(color);
}

//______________________________________________________________________________
void TH1::SetLabelFont(Style_t font, Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) fXaxis.SetLabelFont(font);
   if (ax == 2) fYaxis.SetLabelFont(font);
   if (ax == 3) fZaxis.SetLabelFont(font);
}

//______________________________________________________________________________
void TH1::SetLabelOffset(Float_t offset, Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) fXaxis.SetLabelOffset(offset);
   if (ax == 2) fYaxis.SetLabelOffset(offset);
   if (ax == 3) fZaxis.SetLabelOffset(offset);
}

//______________________________________________________________________________
void TH1::SetLabelSize(Float_t size, Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) fXaxis.SetLabelSize(size);
   if (ax == 2) fYaxis.SetLabelSize(size);
   if (ax == 3) fZaxis.SetLabelSize(size);
}

//______________________________________________________________________________
void TH1::SetTickLength(Float_t length, Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) fXaxis.SetTickLength(length);
   if (ax == 2) fYaxis.SetTickLength(length);
   if (ax == 3) fZaxis.SetTickLength(length);
}

//______________________________________________________________________________
void TH1::SetTitleOffset(Float_t offset, Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) fXaxis.SetTitleOffset(offset);
   if (ax == 2) fYaxis.SetTitleOffset(offset);
   if (ax == 3) fZaxis.SetTitleOffset(offset);
}

//______________________________________________________________________________
void TH1::SetTitleSize(Float_t size, Option_t *axis)
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) fXaxis.SetTitleSize(size);
   if (ax == 2) fYaxis.SetTitleSize(size);
   if (ax == 3) fZaxis.SetTitleSize(size);
}
