// @(#)root/meta:$Name:  $:$Id: TFunction.h,v 1.7 2003/06/13 16:21:21 rdm Exp $
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
   TString         fMangledName;     //Mangled name as determined by CINT.
   TString         fSignature;       //string containing function signature
   TList          *fMethodArgs;      //list of function arguments

   virtual void    CreateSignature();

public:
   TFunction(G__MethodInfo *info = 0);
   TFunction(const TFunction &orig);
   TFunction& operator=(const TFunction &rhs);
   virtual            ~TFunction();
   virtual TObject    *Clone(const char *newname="") const;
   virtual const char *GetMangledName() const;
   virtual const char *GetPrototype() const;
   const char         *GetSignature();
   const char         *GetReturnTypeName() const;
   TList              *GetListOfMethodArgs();
   Int_t               GetNargs() const;
   Int_t               GetNargsOpt() const;
   void               *InterfaceMethod() const;
   Long_t              Property() const;

   ClassDef(TFunction,0)  //Dictionary for global function
};

#endif
