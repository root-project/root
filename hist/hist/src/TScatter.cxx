// @(#)root/hist:$Id$
// Author: Olivier Couet   18/05/2022

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TROOT.h"
#include "TBuffer.h"
#include "TScatter.h"
#include "TStyle.h"
#include "TMath.h"
#include "TVirtualPad.h"
#include "TH1.h"
#include "TF1.h"
#include "TVectorD.h"
#include "TSystem.h"
#include "strtok.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>

ClassImp(TScatter);


////////////////////////////////////////////////////////////////////////////////

/** \class TScatter
    \ingroup Graphs
A TScatter is a TGraph able to draw four variables on a single plot. The two first
variables are the x and y position of the markers and the 3rd is mapped on the current
color map and the 4th on the marker size.

The following example demonstrates how it works:

Begin_Macro(source)
../../../tutorials/graphs/scatter.C
End_Macro

*/


////////////////////////////////////////////////////////////////////////////////
/// TScatter default constructor.

TScatter::TScatter(): TGraph()
{
   if (!CtorAllocate()) return;
   fScale  = 5.;
   fMargin = 0.1;
   fSize   = nullptr;
   fColor  = nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// TScatter normal constructor.
///
///  the arrays are preset to zero

TScatter::TScatter(Int_t n)
   : TGraph(n)
{
   if (!CtorAllocate()) return;
   FillZero(0, fNpoints);
   fScale  = 5.;
   fMargin = 0.1;
   fSize   = nullptr;
   fColor  = nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// TScatter normal constructor.
///
///  if ex or ey are null, the corresponding arrays are preset to zero

TScatter::TScatter(Int_t n, const Double_t *x, const Double_t *y, const Double_t *col, const Double_t *size)
   : TGraph(n, x, y)
{
   if (!CtorAllocate()) return;

   n = sizeof(Double_t) * fNpoints;
   if (col) memcpy(fColor, col, n);
   else     fColor = nullptr;
   if (size) memcpy(fSize, size, n);
   else      fSize = nullptr;
   fScale  = 5.;
   fMargin = 0.1;
}


////////////////////////////////////////////////////////////////////////////////
/// TScatter default destructor.

TScatter::~TScatter()
{
   delete [] fColor;
   delete [] fSize;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor allocate.
///
/// Note: This function should be called only from the constructor
/// since it does not delete previously existing arrays.

Bool_t TScatter::CtorAllocate()
{

   if (!fNpoints) {
      fColor = fSize = nullptr;
      return kFALSE;
   } else {
      fColor = new Double_t[fMaxSize];
      fSize = new Double_t[fMaxSize];
   }
   return kTRUE;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns a pointer to the histogram used to draw the axis

TH1F *TScatter::GetHistogram() const
{
   if (!fHistogram) {
      // do not add the histogram to gDirectory
      // use local TDirectory::TContect that will set temporarly gDirectory to a nullptr and
      // will avoid that histogram is added in the global directory
      {
         TDirectory::TContext ctx(nullptr);
         double rwxmin, rwymin, rwxmax, rwymax;
         int npt = 100;
         ComputeRange(rwxmin, rwymin, rwxmax, rwymax);
         double dx = (rwxmax-rwxmin)*fMargin;
         double dy = (rwymax-rwymin)*fMargin;
         auto h = new TH1F(TString::Format("%s_h",GetName()),GetTitle(),npt,rwxmin,rwxmax);
         h->SetMinimum(rwymin-dy);
         h->SetMaximum(rwymax+dy);
         h->GetXaxis()->SetLimits(rwxmin-dx,rwxmax+dx);
         h->SetBit(TH1::kNoStats);
         h->SetDirectory(0);
         h->Sumw2(kFALSE);
         ((TScatter*)this)->fHistogram = h;//new TH1F(gname, GetTitle(), npt, rwxmin, rwxmax);
      }
   }
   return fHistogram;
}


////////////////////////////////////////////////////////////////////////////////
/// Set zero values for point arrays in the range `[begin, end]`.

void TScatter::FillZero(Int_t begin, Int_t end, Bool_t from_ctor)
{
   if (!from_ctor) {
      TGraph::FillZero(begin, end, from_ctor);
   }
   Int_t n = (end - begin) * sizeof(Double_t);
   memset(fColor + begin, 0, n);
   memset(fSize + begin, 0, n);
}


////////////////////////////////////////////////////////////////////////////////
/// Print graph and errors values.

void TScatter::Print(Option_t *) const
{
   for (Int_t i = 0; i < fNpoints; i++) {
      printf("x[%d]=%g, y[%d]=%g, color[%d]=%g, size[%d]=%g\n", i, fX[i], i, fY[i], i, fColor[i], i, fSize[i]);
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Set the margin around the plot in %

void TScatter::SetMargin(Double_t margin)
{
   if (fMargin != margin) {
      delete fHistogram;
      fHistogram = nullptr;
      fMargin = margin;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TScatter::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   char quote = '"';
   out << "   " << std::endl;
   static Int_t frameNumber = 1000;
   frameNumber++;

   Int_t i;
   TString fXName     = TString::Format("%s_fx%d",GetName(),frameNumber);
   TString fYName     = TString::Format("%s_fy%d", GetName(),frameNumber);
   TString fColorName = TString::Format("%s_fcolor%d",GetName(),frameNumber);
   TString fSizeName  = TString::Format("%s_fsize%d",GetName(),frameNumber);
   out << "   Double_t " << fXName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << fX[i] << "," << std::endl;
   out << "   " << fX[fNpoints-1] << "};" << std::endl;
   out << "   Double_t " << fYName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << fY[i] << "," << std::endl;
   out << "   " << fY[fNpoints-1] << "};" << std::endl;
   out << "   Double_t " << fColorName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << fColor[i] << "," << std::endl;
   out << "   " << fColor[fNpoints-1] << "};" << std::endl;
   out << "   Double_t " << fSizeName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << fSize[i] << "," << std::endl;
   out << "   " << fSize[fNpoints-1] << "};" << std::endl;

   if (gROOT->ClassSaved(TScatter::Class())) out << "   ";
   else out << "   TScatter *";
   out << "scat = new TScatter(" << fNpoints << ","
                                    << fXName   << ","  << fYName  << ","
                                    << fColorName  << ","  << fSizeName << ");"
                                    << std::endl;

   out << "   scat->SetName(" << quote << GetName() << quote << ");" << std::endl;
   out << "   scat->SetTitle(" << quote << GetTitle() << quote << ");" << std::endl;

   SaveFillAttributes(out, "scat", 0, 1001);
   SaveLineAttributes(out, "scat", 1, 1, 1);
   SaveMarkerAttributes(out, "scat", 1, 1, 1);

   if (fHistogram) {
      TString hname = fHistogram->GetName();
      hname += frameNumber;
      fHistogram->SetName(TString::Format("Graph_%s", hname.Data()));
      fHistogram->SavePrimitive(out, "nodraw");
      out << "   scat->SetHistogram(" << fHistogram->GetName() << ");" << std::endl;
      out << "   " << std::endl;
   }

   out << "   scat->Draw(" << quote << option << quote << ");" << std::endl;
}
