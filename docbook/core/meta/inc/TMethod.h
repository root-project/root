// @(#)root/meta:$Id$
// Author: Rene Brun   09/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMethod
#define ROOT_TMethod


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMethod                                                              //
//                                                                      //
// Dictionary of a member function (method).                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFunction
#include "TFunction.h"
#endif

class TList;
class TDataMember;
class TMethodCall;
class TClass;

enum EMenuItemKind {
   kMenuNoMenu = 0,
   kMenuDialog,
   kMenuToggle,
   kMenuSubMenu
};

class TMethod : public TFunction {

private:
   TClass                 *fClass;           //pointer to the class
   EMenuItemKind           fMenuItem;        //type of menuitem in context menu
   TString                 fGetter;          //state getter in case this is a *TOGGLE method
   TMethodCall            *fGetterMethod;    //methodcall for state getter in case this is a *TOGGLE method
   TMethodCall            *fSetterMethod;    //methodcall for state setter in case this is a *TOGGLE method

   void                    CreateSignature();

public:
                           TMethod(MethodInfo_t *info = 0, TClass *cl = 0);
                           TMethod(const TMethod &org);
   TMethod&                operator=(const TMethod &rhs);
   virtual                ~TMethod();
   virtual TObject        *Clone(const char *newname="") const;
   TClass                 *GetClass() const { return fClass; }
   EMenuItemKind           IsMenuItem() const { return fMenuItem; }
   virtual const char     *GetCommentString();
   virtual const char     *Getter() const { return fGetter; }
   virtual TMethodCall    *GetterMethod();
   virtual TMethodCall    *SetterMethod();
   virtual TDataMember    *FindDataMember();
   virtual TList          *GetListOfMethodArgs();
   virtual void            SetMenuItem(EMenuItemKind menuItem) {fMenuItem=menuItem;}

   ClassDef(TMethod,0)  //Dictionary for a class member function (method)
};

#endif
