// @(#)root/meta:$Name:  $:$Id: TMethod.h,v 1.3 2003/06/13 14:21:26 brun Exp $
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
#ifndef ROOT_TString
#include "TString.h"
#endif

class TList;
class TDataMember;
class TClass;
class G__MethodInfo;

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

   void                    CreateSignature();

public:
                           TMethod(G__MethodInfo *info = 0, TClass *cl = 0);
                           TMethod(const TMethod &org);
   TMethod&                operator=(const TMethod &rhs);
   virtual                ~TMethod() { }
   virtual TObject        *Clone(const char *newname="") const;
   TClass                 *GetClass() const { return fClass; }
   EMenuItemKind           IsMenuItem() const { return fMenuItem; }
   virtual const char     *GetCommentString();
   virtual TDataMember    *FindDataMember();
   virtual TList          *GetListOfMethodArgs();
   virtual void            SetMenuItem(EMenuItemKind menuItem) {fMenuItem=menuItem;}

   ClassDef(TMethod,0)  //Dictionary for a class member function (method)
};

#endif
