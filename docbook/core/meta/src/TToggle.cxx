// @(#)root/meta:$Id$
// Author: Piotr Golonka   30/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TToggle                                                              //
//                                                                      //
// This class defines toggling facility for both - object's method or   //
// variables.                                                           //
// Assume that user provides an object with a two-state field , and     //
// methods to Get/Set value of this field. This object enables to switch//
// values via this method when the only thing you know about the field  //
// is the name of the method (or method itself) which sets the field.   //
// This facility is required in context Pop-Up menu, when the only      //
// information about how to toggle a field is a name of methhod which   //
// sets it.                                                             //
// This class may be also used for toggling an integer variable,        //
// which may be important while building universal objects...           //
// When user provides a "set-method" of name SetXXX this object tries   //
// automaticaly find a matching "get-method" by lookin for a method     //
// with name GetXXX, IsXXX or HasXXX for given object.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMethod.h"
#include "TToggle.h"
#include "TDataMember.h"


ClassImp(TToggle)

//______________________________________________________________________________
TToggle::TToggle()
{
   // TToggle default constructor. You have to initialize it before using
   // by making a call to SetToggledVariable() or SetToggledObject().

   fState       =  kFALSE;
   fValue       = -1;
   fOnValue     =  1;
   fOffValue    =  0;
   fInitialized =  0;
   fObject      =  0;
   fGetter      =  0;
   fSetter      =  0;
   fTglVariable =  0;
}

//______________________________________________________________________________
void TToggle::SetToggledVariable(Int_t &var)
{
   // Initializes object for use with a variable - you pass it via reference
   // so it will be modified by Toggle.

   fTglVariable=&var;
   fValue=var;
}

//______________________________________________________________________________
Bool_t TToggle::GetState()
{
   // Returns the state of Toggle according to its current value and
   // fOnValue, returns true if they match.

   if (fInitialized)
      if (fGetter) fGetter->Execute(fObject, fValue);
   return (fState = (fValue == fOnValue));
}

//______________________________________________________________________________
void TToggle::SetState(Bool_t state)
{
   // Sets the value of toggle to fOnValue or fOffValue according to passed
   // argument.

   if (fInitialized) {
      char stringon[20];
      char stringoff[20];
      snprintf(stringon,sizeof(stringon),"%li",fOnValue);
      snprintf(stringoff,sizeof(stringoff),"%li",fOffValue);

      fSetter->Execute(fObject, state ? stringon:stringoff);
      fState=state;
      fValue= ( state ? fOnValue : fOffValue);
   }
}

//______________________________________________________________________________
void TToggle::SetValue(Long_t val)
{
   // Sets the value of toggle and modifies its state according to whether
   // the value is equal to fOnValue.

   if (fInitialized) {
      char stringval[20];
      snprintf(stringval,sizeof(stringval),"%li",val);
      fSetter->Execute(fObject, stringval);
      fState=(val==fOnValue);
      fValue= val;
   }
}

//______________________________________________________________________________
void TToggle::Toggle()
{
   // Toggles the Values and State of this object and connected data!

   if (fInitialized){
      if (fTglVariable){
         *fTglVariable = !(*fTglVariable);
         fValue=(*fTglVariable);
         fState=*fTglVariable;
      }
      if (fGetter && fSetter){
         fGetter->Execute(fObject,fValue);
         fValue=( (fValue==fOnValue) ? fOffValue:fOnValue);
         fState=(!(fValue!=fOnValue));
         char stringon[20];
         snprintf(stringon,sizeof(stringon),"%li",fValue);
         fSetter->Execute(fObject, stringon);
      }
   }
}

//______________________________________________________________________________
void TToggle::SetToggledObject(TObject *obj, TMethod *anymethod)
{
   // Initializes it to toggle an object's datamember using this object's
   // method.

   fObject = obj;
   TDataMember *m = anymethod->FindDataMember();
   if (!m) {
      // try to see if the TMethod has a getter associated via the *GETTER=
      // comment string
      if (anymethod->GetterMethod()) {
         fGetter = anymethod->GetterMethod();
         fSetter = anymethod->SetterMethod();
         fInitialized = 1;
      } else
         Error("SetToggledObject", "cannot determine getter method for %s", anymethod->GetName());
   } else {
      fGetter = m->GetterMethod(obj->IsA());
      fSetter = m->SetterMethod(obj->IsA());
      fInitialized = 1;
   }
}
