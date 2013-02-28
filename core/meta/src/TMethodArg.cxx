// @(#)root/meta:$Id$
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
TMethodArg::TMethodArg(MethodArgInfo_t *info, TFunction *method) : TDictionary()
{
   // Default TMethodArg ctor. TMethodArgs are constructed in TFunction
   // via a call to TCint::CreateListOfMethodArgs().

   fDataMember = 0;
   fInfo       = info;
   fMethod     = method;
   if (fInfo) {
      SetName(gCint->MethodArgInfo_Name(fInfo));
      SetTitle(gCint->MethodArgInfo_TypeName(fInfo));
   }
}

//______________________________________________________________________________
TMethodArg::~TMethodArg()
{
   // TMethodArg dtor deletes adopted CINT MethodArgInfo object.

   if (fInfo) gCint->MethodArgInfo_Delete(fInfo);
}

//______________________________________________________________________________
const char *TMethodArg::GetDefault() const
{
   // Get default value of method argument.

   return gCint->MethodArgInfo_DefaultValue(fInfo);
}

//______________________________________________________________________________
const char *TMethodArg::GetTypeName() const
{
   // Get type of method argument, e.g.: "class TDirectory*" -> "TDirectory"
   // Result needs to be used or copied immediately.

   return gCint->TypeName(gCint->MethodArgInfo_TypeName(fInfo));
}

//______________________________________________________________________________
const char *TMethodArg::GetFullTypeName() const
{
   // Get full type description of method argument, e.g.: "class TDirectory*".
   
   return gCint->MethodArgInfo_TypeName(fInfo);
}

//______________________________________________________________________________
std::string TMethodArg::GetTypeNormalizedName() const
{
   // Get the normalized name of the return type.  A normalized name is fully
   // qualified and has all typedef desugared except for the 'special' typedef
   // which include Double32_t, Float16_t, [U]Long64_t and std::string.  It
   // also has std:: removed [This is subject to change].
   //
   
   return gCint->MethodArgInfo_TypeNormalizedName(fInfo);
}

//______________________________________________________________________________
Long_t TMethodArg::Property() const
{
   // Get property description word. For meaning of bits see EProperty.

   return gCint->MethodArgInfo_Property(fInfo);
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

