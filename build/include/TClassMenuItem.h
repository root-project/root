// @(#)root/meta:$Id$
// Author: Damir Buskulic   23/11/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClassMenuItem
#define ROOT_TClassMenuItem


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClassMenuItem                                                       //
//                                                                      //
// Describe one element of the context menu associated to a class       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"

class TList;
class TClass;


class TClassMenuItem : public TObject {

public:
   enum EClassMenuItemType {
      kPopupUserFunction, kPopupSeparator, kPopupStandardList
   };
   enum { kIsExternal, kIsSelf };

private:
   EClassMenuItemType  fType;          //type flag (EClassMenuItemType)
   Int_t               fSelfObjectPos; //rang in argument list corresponding to the object being clicked on
   Bool_t              fSelf;          //flag to indicate that object to be called is the selected one
   Bool_t              fToggle;        //flag toggle method
   TString             fTitle;         //title if not standard
   TObject            *fCalledObject;  //object to be called
   TString             fFunctionName;  //name of the function or method to be called
   TString             fArgs;          //arguments type list *** NOT CHECKED ***
   TList              *fSubMenu;       //list of submenu items
   TClass             *fParent;        //parent class

protected:
   TClassMenuItem(const TClassMenuItem&);
   TClassMenuItem& operator=(const TClassMenuItem&);

public:
   TClassMenuItem();
   TClassMenuItem(Int_t type, TClass *parent, const char *title="",
                  const char *functionname="", TObject *obj=nullptr,
                  const char *args="", Int_t selfobjposition=-1,
                  Bool_t self=kFALSE);
   virtual        ~TClassMenuItem();
   const char    *GetTitle() const override { return fTitle.Data(); }
   virtual const char *GetFunctionName() const { return fFunctionName; }
   virtual const char *GetArgs() const { return fArgs; }
   virtual TObject *GetCalledObject() const { return fCalledObject; }
   virtual Int_t    GetType() const { return fType; }
   virtual Int_t    GetSelfObjectPos() const { return fSelfObjectPos; }
   virtual Bool_t   IsCallSelf() const { return fSelf; }
   virtual Bool_t   IsSeparator() const { return fType==kPopupSeparator ? kTRUE : kFALSE; }
   virtual Bool_t   IsStandardList() const { return fType==kPopupStandardList ? kTRUE : kFALSE; }
   virtual Bool_t   IsToggle() const { return fToggle; }
   virtual void     SetType(Int_t type) { fType = (EClassMenuItemType) type; }
   virtual void     SetTitle(const char *title) { fTitle = title; }
   virtual void     SetSelf(Bool_t self) { fSelf = self; }
   virtual void     SetToggle(Bool_t toggle = kTRUE) { fToggle = toggle; }
   virtual void     SetCall(TObject *obj, const char *method,
                            const char *args="", Int_t selfobjposition = 0)
                       { fCalledObject = obj; fFunctionName = method;
                         fArgs = args; fSelfObjectPos = selfobjposition;}

   ClassDefOverride(TClassMenuItem,0)  //One element of the class context menu
};

#endif
