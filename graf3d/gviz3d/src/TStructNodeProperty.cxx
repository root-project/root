// @(#)root/gviz3d:$Id$
// Author: Tomasz Sosnicki   18/09/09

/************************************************************************
* Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include "TStructNodeProperty.h"
#include <TROOT.h>
#include <TClass.h>

ClassImp(TStructNodeProperty);

//________________________________________________________________________
//////////////////////////////////////////////////////////////////////////
//
// TStructNodeProperty class keeps a color for type
//
//
//
//
//
//////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Contructs a TStructNodeProperty with p as parent window for class "name" with color "color".

TStructNodeProperty::TStructNodeProperty(const char * name, Int_t color)
{
   SetName(name);
   SetColor(color);
}

////////////////////////////////////////////////////////////////////////////////
/// Contructs a TStructNodeProperty with p as parent window for class "name" with color "pixel".

TStructNodeProperty::TStructNodeProperty(const char * name, Pixel_t pixel)
{
   SetName(name);
   SetColor(pixel);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TStructNodeProperty::~TStructNodeProperty()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Overrided method. Compare two objects of TStructNodeProperty class

Int_t TStructNodeProperty::Compare(const TObject* obj) const
{

   // Default color "+" should be at the end.
   if (fName == "+") {
      return 1;
   }
   TStructNodeProperty* prop = (TStructNodeProperty*)obj;
   TString propName(prop->GetName());
   if (propName == "+") {
      return -1;
   }

   TClass* cl1;
   if (fName.EndsWith("+")) {
      cl1 = TClass::GetClass(TString(fName.Data(), fName.Length()-1).Data());
   } else {
      cl1 = TClass::GetClass(fName.Data());
   }

   TClass* cl2;

   if (propName.EndsWith("+")) {
      cl2 = TClass::GetClass(TString(propName.Data(), propName.Length()-1).Data());
   } else {
      cl2 = TClass::GetClass(prop->GetName());
   }

   if(!cl1) {
      return -1;
   }
   if(!cl2) {
      return -1;
   }

   if(cl1->InheritsFrom(cl2)) {
      return -1;
   }
   if(cl2->InheritsFrom(cl1)) {
      return 1;
   }

   if(this > prop) {
      return 1;
   }
   if(this < prop) {
      return -1;
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns color of class

TColor TStructNodeProperty::GetColor() const
{
   return fColor;
}

////////////////////////////////////////////////////////////////////////////////
/// Return color in Pixel_t format

Pixel_t TStructNodeProperty::GetPixel() const
{
   return fColor.GetPixel();
}

////////////////////////////////////////////////////////////////////////////////
/// Retruns true, because we have overrided method Compare

Bool_t TStructNodeProperty::IsSortable() const
{
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the color to "color"

void TStructNodeProperty::SetColor(const TColor & color)
{
   fColor = color;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets  the color to "color"

void TStructNodeProperty::SetColor(Int_t color)
{
   fColor = *(gROOT->GetColor(color));
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the color to "pixel"

void TStructNodeProperty::SetColor(Pixel_t pixel)
{
   SetColor(TColor::GetColor(pixel));
}
