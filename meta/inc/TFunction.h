// @(#)root/meta:$Name:  $:$Id: TFunction.h,v 1.1.1.1 2000/05/16 17:00:43 rdm Exp $
// Author: Fons Rademakers   07/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFunction
#define ROOT_TFunction


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFunction                                                            //
//                                                                      //
// Dictionary of global functions (global functions are obtained from   //
// CINT).                                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class G__MethodInfo;


class TFunction : public TDictionary {

friend class TCint;

protected:
   G__MethodInfo  *fInfo;            //pointer to CINT function info
   TString         fSignature;       //string containing function signature
   TList          *fMethodArgs;      //list of function arguments

   virtual void    CreateSignature();

public:
   TFunction(G__MethodInfo *info = 0);
   virtual     ~TFunction();
   Int_t        Compare(const TObject *obj) const;
   const char  *GetName() const;
   const char  *GetSignature();
   const char  *GetTitle() const;
   const char  *GetReturnTypeName() const;
   TList       *GetListOfMethodArgs();
   Int_t        GetNargs() const;
   Int_t        GetNargsOpt() const;
   ULong_t      Hash() const;
   void        *InterfaceMethod() const;
   Long_t       Property() const;

   ClassDef(TFunction,0)  //Dictionary for global function
};

#endif
