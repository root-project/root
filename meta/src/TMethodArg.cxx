// @(#)root/meta:$Name:  $:$Id: TMethodArg.cxx,v 1.1.1.1 2000/05/16 17:00:43 rdm Exp $
// Author: Rene Brun   04/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMethodArg.h"
#include "TROOT.h"
#include "TInterpreter.h"
#include "Strlen.h"
#include "Api.h"
#include "TMethod.h"
#include "TMethodCall.h"
#include "TDataMember.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Each ROOT method (see TMethod) has a linked list of its arguments.  //
//  This class describes one single method argument.                    //
//  The method argument info is obtained via the CINT api.              //
//  See class TCint.                                                    //
//                                                                      //
//  The method argument information is used a.o. in the TContextMenu    //
//  and THtml classes.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TMethodArg)

//______________________________________________________________________________
TMethodArg::TMethodArg(G__MethodArgInfo *info, TFunction *method) : TDictionary()
{
   // Default TMethodArg ctor. TMethodArgs are constructed in TFunction
   // via a call to TCint::CreateListOfMethodArgs().

   fDataMember = 0;
   fInfo       = info;
   fMethod     = method;
}

//______________________________________________________________________________
TMethodArg::~TMethodArg()
{
   // TMethodArg dtor deletes adopted G__MethodArgInfo object.

   if (fInfo) delete fInfo;
}

//______________________________________________________________________________
const char *TMethodArg::GetDefault() const
{
   // Get default value of method argument.

   return fInfo->DefaultValue();
}

//______________________________________________________________________________
const char *TMethodArg::GetName() const
{
   // Get method argument name (can be empty string or 0).

   return fInfo->Name();
}

//______________________________________________________________________________
const char *TMethodArg::GetTypeName() const
{
   // Get type of method argument, e.g.: "class TDirectory*" -> "TDirectory"
   // Result needs to be used or copied immediately.

   return gInterpreter->TypeName(fInfo->Type()->Name());
}

//______________________________________________________________________________
const char *TMethodArg::GetFullTypeName() const
{
   // Get full type description of method argument, e.g.: "class TDirectory*".

   return fInfo->Type()->Name();
}

//______________________________________________________________________________
const char *TMethodArg::GetTitle() const
{
   // Get method argument description. This method always returns 0.
   // However, for backward compatibility, we currently return
   // GetFullTypeName().

   return GetFullTypeName();
}

//______________________________________________________________________________
Int_t TMethodArg::Compare(const TObject *obj) const
{
   // Compare to other object. Returns 0<, 0 or >0 depending on
   // whether "this" is lexicographically less than, equal to, or
   // greater than obj.

   if (fInfo->Name())
      return strcmp(fInfo->Name(), obj->GetName());
   else return -1;
}

//______________________________________________________________________________
ULong_t TMethodArg::Hash() const
{
   // Return hash value for TDataType based on its name.

   if (fInfo->Name()) {
      TString s = fInfo->Name();
      return s.Hash();
   } else return 0;
}

//______________________________________________________________________________
Long_t TMethodArg::Property() const
{
   // Get property description word. For meaning of bits see EProperty.

   return fInfo->Property();
}

//______________________________________________________________________________
TList *TMethodArg::GetOptions() const
{
   // Returns list of possible options - according to pointed datamember.
   // If there is no datamember field assigned to this methodarg - returns 0.

   return (TList*)(fDataMember ? fDataMember->GetOptions() : 0);
}

//______________________________________________________________________________
TDataMember *TMethodArg::GetDataMember() const
{
   // Returns TDataMember pointed by this methodarg.
   // If you want to specify list of options or current value for your
   // MethodArg (i.e. it is used as initial values in argument-asking dialogs
   // popped up from context-meny),you can get this value from one of data
   // members of the class.
   // The only restriction is, that this DataMember object must have its
   // Getter/Setter methods set-up correctly - for details look at TDataMember.
   // To learn how to specify the data member to which the argument should
   // "point", look at TMethod. This is TMethod which sets up fDataMember,
   // so it could work correctly.

   return fDataMember;
}

