// @(#)root/meta:$Name:  $:$Id: TMethodArg.h,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
// Author: Rene Brun   04/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMethodArg
#define ROOT_TMethodArg


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMethodArg                                                           //
//                                                                      //
// Dictionary interface for a method argument.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif

#include "TDataMember.h"

class TFunction;
class TMethod;
class G__MethodArgInfo;


class TMethodArg : public TDictionary {

friend class TMethod;

private:
   G__MethodArgInfo  *fInfo;         //pointer to CINT method argument info
   TFunction         *fMethod;       //pointer to the method or global function
   TDataMember       *fDataMember;   //TDataMember pointed by this arg,to get values and options from.

public:
   TMethodArg(G__MethodArgInfo *info = 0, TFunction *method = 0);
   virtual       ~TMethodArg();
   Int_t          Compare(const TObject *obj) const;
   const char    *GetDefault() const;
   TFunction     *GetMethod() const { return fMethod; }
   const char    *GetName() const;
   const char    *GetTitle() const;
   const char    *GetTypeName() const;
   const char    *GetFullTypeName() const;
   ULong_t        Hash() const;
   Long_t         Property() const;

   TDataMember   *GetDataMember() const;
   TList         *GetOptions() const;

   ClassDef(TMethodArg,0)  //Dictionary for a method argument
};

#endif

