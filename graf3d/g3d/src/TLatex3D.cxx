// @(#)root/g3d:$Id$
// Author: Olivier Couet   03/05/23

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TLatex3D.h"
#include "TVirtualPad.h"
#include "TView.h"
#include "TVirtualViewer3D.h"


ClassImp(TLatex3D);

/** \class TLatex3D
\ingroup g3d
A 3-dimensional TLatex.
Example:
Begin_Macro(source)
../../../tutorials/graphs/annotation3d.C
End_Macro
*/

////////////////////////////////////////////////////////////////////////////////
/// 3-D text  default constructor.

TLatex3D::TLatex3D(Double_t x, Double_t y, Double_t z, const char *text)
{
   fX3D = x;
   fY3D = y;
   fZ3D = z;
   fTitle = text;
}


////////////////////////////////////////////////////////////////////////////////
/// 3-D text default destructor.


TLatex3D::~TLatex3D()
{
}

////////////////////////////////////////////////////////////////////////////////
/// List this 3-D text with its attributes.

void TLatex3D::ls(Option_t *) const
{
   TROOT::IndentLevel();
   printf("OBJ: %s\t%s  \tX= %f Y=%f Z=%f \n",IsA()->GetName(),GetTitle(),fX3D,fY3D,fZ3D);
}


////////////////////////////////////////////////////////////////////////////////
/// Draw this text with new coordinates.

TLatex3D *TLatex3D::DrawText3D(Double_t x, Double_t y, Double_t z, const char *text)
{
   TLatex3D *newtext = new TLatex3D(x, y, z, text);
   TAttText::Copy(*newtext);
   newtext->SetBit(kCanDelete);
   if (TestBit(kTextNDC)) newtext->SetNDC();
   newtext->AppendPad();
   return newtext;
}


////////////////////////////////////////////////////////////////////////////////
/// Paint a TLatex3D.

void TLatex3D::Paint(Option_t * /* option */ )
{
   TView *view = gPad->GetView();
   if (!view) {
      PaintLatex(fX3D,fY3D,GetTextAngle(),GetTextSize(),GetTitle());
   } else {
      double xyz[3];
      xyz[0] = fX3D;
      xyz[1] = fY3D;
      xyz[2] = fZ3D;
      Double_t xpad[3];
      view->WCtoNDC(xyz, &xpad[0]);
      PaintLatex(xpad[0],xpad[1],GetTextAngle(),GetTextSize(),GetTitle());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Dump this text with its attributes.

void TLatex3D::Print(Option_t *) const
{
   printf("Text  X=%f Y=%f Z = %f Text=%s Font=%d Size=%f",fX3D,fY3D,fZ3D,GetTitle(),GetTextFont(),GetTextSize());
   if (GetTextColor() != 1 ) printf(" Color=%d",GetTextColor());
   if (GetTextAlign() != 10) printf(" Align=%d",GetTextAlign());
   if (GetTextAngle() != 0 ) printf(" Angle=%f",GetTextAngle());
   printf("\n");
}
