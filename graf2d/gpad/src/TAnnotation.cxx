// @(#)root/gpad:$Id$
// Author: Olivier Couet   03/05/23

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TAnnotation.h"
#include "TVirtualPad.h"
#include "TView.h"
#include "TVirtualViewer3D.h"


ClassImp(TAnnotation);

/** \class TAnnotation
\ingroup gpad

An annotation is a TLatex which can be drawn in a 2D or 3D space.

Example:

Begin_Macro(source)
{
   auto hsurf1 = new TH2F("hsurf1","3D text example ",30,-4,4,30,-20,20);
   float px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hsurf1->Fill(px-1,5*py);
      hsurf1->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hsurf1->Draw("SURF1");
   int binx,biny,binz;
   int bmax = hsurf1->GetMaximumBin(binx,biny,binz);
   double xm = hsurf1->GetXaxis()->GetBinCenter(binx);
   double ym = hsurf1->GetYaxis()->GetBinCenter(biny);
   double zm = hsurf1->GetMaximum();
   auto t = new TAnnotation(xm,ym,zm,Form("Maximum = %g",zm));
   t->SetTextFont(42);
   t->SetTextSize(0.03);
   t->Draw();
}
End_Macro

Another example:

Begin_Macro(source)
../../../tutorials/visualisation/graphics/annotation3d.C
End_Macro
*/

////////////////////////////////////////////////////////////////////////////////
/// annotation default constructor.

TAnnotation::TAnnotation(Double_t x, Double_t y, Double_t z, const char *text)
{
   fX = x;
   fY = y;
   fZ = z;
   fTitle = text;
}


////////////////////////////////////////////////////////////////////////////////
/// annotation default destructor.


TAnnotation::~TAnnotation()
{
}

////////////////////////////////////////////////////////////////////////////////
/// List this annotation with its attributes.

void TAnnotation::ls(Option_t *) const
{
   TROOT::IndentLevel();
   printf("OBJ: %s\t%s  \tX= %f Y=%f Z=%f \n",IsA()->GetName(),GetTitle(),fX,fY,fZ);
}


////////////////////////////////////////////////////////////////////////////////
/// Draw this annotation with new coordinates.

TAnnotation *TAnnotation::DrawAnnotation(Double_t x, Double_t y, Double_t z, const char *text)
{
   TAnnotation *newannotation = new TAnnotation(x, y, z, text);
   TAttText::Copy(*newannotation);
   newannotation->SetBit(kCanDelete);
   if (TestBit(kTextNDC)) newannotation->SetNDC();
   newannotation->AppendPad();
   return newannotation;
}


////////////////////////////////////////////////////////////////////////////////
/// Paint this annotation with new coordinates.

void TAnnotation::PaintAnnotation(Double_t x, Double_t y, Double_t z, Double_t angle, Double_t size, const Char_t *text)
{
   TView *view = gPad->GetView();
   if (!view) {
      PaintLatex(x, y, angle, size, text);
   } else {
      Double_t xyz[3] = { x, y, z }, xpad[3];
      view->WCtoNDC(xyz, &xpad[0]);
      PaintLatex(xpad[0], xpad[1], angle, size, text);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint a TAnnotation.

void TAnnotation::Paint(Option_t * /* option */ )
{
   PaintAnnotation(fX, fY, fZ, GetTextAngle(),GetTextSize(),GetTitle());
}

////////////////////////////////////////////////////////////////////////////////
/// Dump this annotation with its attributes.

void TAnnotation::Print(Option_t *) const
{
   printf("Annotation  X=%f Y=%f Z = %f Text=%s Font=%d Size=%f",fX,fY,fZ,GetTitle(),GetTextFont(),GetTextSize());
   if (GetTextColor() != 1 ) printf(" Color=%d",GetTextColor());
   if (GetTextAlign() != 10) printf(" Align=%d",GetTextAlign());
   if (GetTextAngle() != 0 ) printf(" Angle=%f",GetTextAngle());
   printf("\n");
}
