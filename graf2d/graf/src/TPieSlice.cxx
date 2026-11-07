// @(#)root/graf:$Id$
// Author: Guido Volpi, Olivier Couet 03/11/2006
// Author: Sergey Linev 09/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPieSlice.h"

#include "TError.h"
#include "TMath.h"
#include "TVirtualPad.h"
#include "TPie.h"

#include <iostream>
#include <cstring>


/** \class TPieSlice
\ingroup BasicGraphics

A slice of a piechart, see the TPie class.

This class describe the property of single
*/

////////////////////////////////////////////////////////////////////////////////
/// This is the default constructor, used to create the standard.

TPieSlice::TPieSlice()
{
   fPie = nullptr;
   fValue = 1;
}

////////////////////////////////////////////////////////////////////////////////
/// This constructor create a slice with a particular value.

TPieSlice::TPieSlice(const char *name, const char *title,
                     TPie *pie, Double_t val) :
                     TNamed(name, title)
{
   fPie = pie;
   fValue = val;
}

////////////////////////////////////////////////////////////////////////////////
/// Eval if the mouse is over the area associated with this slice.
/// Return 0 only when mouse over the outer part of the slice to let activate context menu

Int_t TPieSlice::DistancetoPrimitive(Int_t px, Int_t py)
{
   Int_t dist = 9999;

   if (gPad && fPie) {
      auto info = fPie->FindSlice(*gPad, px, py);
      if ((info.num >= 0) && (fPie->GetSlice(info.num) == this) && (info.rad > 0.6) && (info.rad <= 1))
         dist = 0;
   }

   return dist;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute event,
/// redirect to TPie object

void TPieSlice::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   if (fPie)
      fPie->ExecuteEvent(event, px, py);
}

////////////////////////////////////////////////////////////////////////////////
/// return the value of the offset in radial direction for this slice.

Double_t TPieSlice::GetRadiusOffset() const
{
   return fRadiusOffset;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the value of this slice.

Double_t TPieSlice::GetValue() const
{
   return fValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Save as C++ macro, used directly from TPie

void TPieSlice::SavePrimitive(std::ostream &out, Option_t *opts)
{
   const char *name = opts;
   if (!name || !*name || strncmp(name, "pie->", 5))
      return;

   out << "   " << name << "->SetTitle(\"" << GetTitle() << "\");\n";
   out << "   " << name << "->SetValue(" << GetValue() << ");\n";
   out << "   " << name << "->SetRadiusOffset(" << GetRadiusOffset() << ");\n";

   SaveFillAttributes(out, name, -1, -1);
   SaveLineAttributes(out, name, 1, 1, 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the radial offset of this slice.

void TPieSlice::SetRadiusOffset(Double_t val)
{
   fRadiusOffset = TMath::Max(val, 0.);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the value for this slice.
/// Negative values are changed with its absolute value.

void TPieSlice::SetValue(Double_t val)
{
   fValue = val;
   if (fValue < .0) {
      Warning("SetValue","Invalid negative value. Absolute value taken");
      fValue *= -1;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy TPieSlice

void TPieSlice::Copy(TObject &obj) const
{
   auto &slice = (TPieSlice&)obj;

   TNamed::Copy(slice);
   TAttLine::Copy(slice);
   TAttFill::Copy(slice);

   slice.SetValue(GetValue());
   slice.SetRadiusOffset(GetRadiusOffset());
}

