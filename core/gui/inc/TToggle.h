// @(#)root/meta:$Id$
// Author: Piotr Golonka   30/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TToggle
#define ROOT_TToggle


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TToggle                                                              //
//                                                                      //
// This class defines toggling facility for both - object's method or   //
// variables.                                                           //
// Assume that user provides an object with a two-state field, and      //
// methods to Get/Set value of this field. This object enables to switch//
// values via this method when the only thing you know about the field  //
// is the name of the method (or method itself) which sets the field.   //
// This facility is required in context popup menu, when the only       //
// information about how to toggle a field is a name of method which    //
// sets it.                                                             //
// This Object may be also used for toggling an integer variable,       //
// which may be important while building universal objects...           //
// When user provides a "set-method" of name SetXXX this object tries   //
// automaticaly to find a matching "get-method" by looking for a method //
// with name GetXXX or IsXXX for given object.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"

#ifdef R__LESS_INCLUDES
class TMethodCall;
class TMethod;
#else
#include "TMethodCall.h"
#include "TMethod.h"
#endif

class TDataMember;

class TToggle: public TNamed {

private:
   Bool_t       fState = kFALSE;    //Object's state - "a local copy"
   Long_t       fOnValue = 1;       //Value recognized as switched ON (Def=1)
   Long_t       fOffValue = 0;      //Value recognized as switched OFF(Def=0)
   Longptr_t    fValue = -1;        //Local copy of a value returned by called function

protected:
   Int_t       fInitialized = 0;    //0 - off, 1 - only object and member is set, 2 - also getters and setters for the object is set
   TObject     *fObject = nullptr;  //The object this Toggle belongs to
   TDataMember *fMember = nullptr;  //Data member which will be toggled
   TMethodCall *fGetter = nullptr;  //Method to Get a value of fObject;
   TMethodCall *fSetter = nullptr;  //Method to Set a value of fObject;

   Int_t       *fTglVariable = nullptr;  //Alternatively: pointer to an integer value to be Toggled instead of TObjectl

public:
   TToggle();
   virtual void    SetToggledObject(TObject *obj, TMethod *anymethod);
   virtual void    SetToggledObject(TObject *obj, TDataMember *member);

   // you just provide any method which has got an initialized pointer
   // to TDataMember... The rest is done automatically...

   virtual void    SetToggledVariable(Int_t &var);

   virtual Bool_t  IsInitialized() { return fInitialized > 0; }

   virtual Bool_t  GetState();
   virtual void    SetState(Bool_t state);
   virtual void    Toggle();

   virtual void    SetOnValue(Long_t lon) { fOnValue=lon; }
   virtual Long_t  GetOnValue() { return fOnValue; }
   virtual void    SetOffValue(Long_t lof) { fOffValue=lof; }
   virtual Long_t  GetOffValue() { return fOffValue; }

   virtual Int_t   GetValue() { return fValue; }
   virtual void    SetValue(Long_t val);

   TMethodCall    *GetGetter() const;
   TMethodCall    *GetSetter() const;

   ClassDefOverride(TToggle,0)  //Facility for toggling datamembers on/off
};

#endif
