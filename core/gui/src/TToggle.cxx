// @(#)root/meta:$Id$
// Author: Piotr Golonka   30/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TToggle

This class defines toggling facility for both - object's method or
variables.
Assume that user provides an object with a two-state field , and
methods to Get/Set value of this field. This object enables to switch
values via this method when the only thing you know about the field
is the name of the method (or method itself) which sets the field.
This facility is required in context Pop-Up menu, when the only
information about how to toggle a field is a name of methhod which
sets it.
This class may be also used for toggling an integer variable,
which may be important while building universal objects...
When user provides a "set-method" of name SetXXX this object tries
automaticaly find a matching "get-method" by lookin for a method
with name GetXXX, IsXXX or HasXXX for given object.
*/

#include "TToggle.h"

#include "TMethod.h"
#include "TMethodCall.h"
#include "TDataMember.h"

#include <cstdio>

////////////////////////////////////////////////////////////////////////////////
/// TToggle default constructor. You have to initialize it before using
/// by making a call to SetToggledVariable() or SetToggledObject().

TToggle::TToggle()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Initializes object for use with a variable - you pass it via reference
/// so it will be modified by Toggle.

void TToggle::SetToggledVariable(Int_t &var)
{
   fTglVariable = &var;
   fValue = var;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the state of Toggle according to its current value and
/// fOnValue, returns true if they match.

Bool_t TToggle::GetState()
{
   if (auto fget = GetGetter())
      fget->Execute(fObject, fValue);

   fState = (fValue == fOnValue);
   return fState;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the value of toggle to fOnValue or fOffValue according to passed
/// argument.

void TToggle::SetState(Bool_t state)
{
   SetValue(state ? fOnValue : fOffValue);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the value of toggle and modifies its state according to whether
/// the value is equal to fOnValue.

void TToggle::SetValue(Long_t val)
{
   if (auto fset = GetSetter()) {
      fset->Execute(fObject, TString::Format("%li", val).Data());
      fState = (val == fOnValue);
      fValue = val;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Toggles the Values and State of this object and connected data!

void TToggle::Toggle()
{
   if (IsInitialized()) {
      if (fTglVariable){
         *fTglVariable = !(*fTglVariable);
         fValue = (*fTglVariable);
         fState = *fTglVariable;
      }
      auto fget = GetGetter();
      auto fset = GetSetter();
      if (fget && fset) {
         fget->Execute(fObject, fValue);
         fValue = (fValue == fOnValue) ? fOffValue : fOnValue;
         fState = (fValue == fOnValue);
         fset->Execute(fObject, TString::Format("%zi", (size_t)fValue).Data());
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initializes it to toggle an object's datamember using this object's method.

void TToggle::SetToggledObject(TObject *obj, TMethod *anymethod)
{
   TDataMember *member = anymethod->FindDataMember();
   if (member)
      TToggle::SetToggledObject(obj, member);
   else if (!anymethod)
      Error("SetToggledObject", "Method pointer not provided");
   else if (!anymethod->GetterMethod())
      Error("SetToggledObject", "cannot determine getter method for %s", anymethod->GetName());
   else {
      // try to see if the TMethod has a getter associated via the *GETTER= comment string
      fObject = obj;
      // TMethodCall pointer can be stored - it is not recerated
      fGetter = anymethod->GetterMethod();
      fSetter = anymethod->SetterMethod();
      fInitialized = 2;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initializes it to toggle an object's datamember

void TToggle::SetToggledObject(TObject *obj, TDataMember *member)
{
   fObject = obj;
   fMember = member;
   fInitialized = 1;

   // one cannot remember TMethodCall pointers from TDataMember
   // while they will be recreated any time they requested again
   fGetter = nullptr;
   fSetter = nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// return getter method
/// request new for the TDataMember while it can be recreated at any time

TMethodCall *TToggle::GetGetter() const
{
   if ((fInitialized == 1) && fMember && fObject)
      return const_cast<TDataMember *>(fMember)->GetterMethod(fObject->IsA());

   return fGetter;
}

////////////////////////////////////////////////////////////////////////////////
/// return setter method
/// request new for the TDataMember while it can be recreated at any time

TMethodCall *TToggle::GetSetter() const
{
   if ((fInitialized == 1) && fMember && fObject)
      return const_cast<TDataMember *>(fMember)->SetterMethod(fObject->IsA());

   return fSetter;
}
