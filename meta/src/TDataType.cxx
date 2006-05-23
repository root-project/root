// @(#)root/meta:$Name:  $:$Id: TDataType.cxx,v 1.21 2005/11/16 20:10:45 pcanal Exp $
// Author: Rene Brun   04/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //                                                                      //
// Basic data type descriptor (datatype information is obtained from    //
// CINT). This class describes the attributes of type definitions       //
// (typedef's). The TROOT class contains a list of all currently        //
// defined types (accessible via TROOT::GetListOfTypes()).              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TDataType.h"
#include "TInterpreter.h"
#include "Api.h"
#ifdef R__SOLARIS
#include <typeinfo>
#endif

ClassImp(TDataType)

//______________________________________________________________________________
TDataType::TDataType(G__TypedefInfo *info) : TDictionary()
{
   // Default TDataType ctor. TDataTypes are constructed in TROOT via
   // a call to TCint::UpdateListOfTypes().

   fInfo = info;

   if (fInfo) {
      SetName(info->Name());
      SetTitle(info->Title());
      SetType(fInfo->TrueName());
      fProperty = fInfo->Property();
      fSize = fInfo->Size();
   } else {
      SetTitle("Builtin basic type");
   }
}

//______________________________________________________________________________
TDataType::TDataType(const char *typenam) : fInfo(0), fProperty(kIsFundamental)
{
   // Constructor for basic data types, like "char", "unsigned char", etc.

   fInfo = 0;
   SetName(typenam);
   SetTitle("Builtin basic type");

   SetType(fName.Data());
}

//______________________________________________________________________________
TDataType::TDataType(const TDataType& dt) :
  TDictionary(dt),
  fInfo(dt.fInfo),
  fSize(dt.fSize),
  fType(dt.fType),
  fProperty(dt.fProperty),
  fTrueName(dt.fTrueName)
{ }

//______________________________________________________________________________
TDataType& TDataType::operator=(const TDataType& dt)
{
  if(this!=&dt) {
    TDictionary::operator=(dt);
    fInfo=dt.fInfo;
    fSize=dt.fSize;
    fType=dt.fType;
    fProperty=dt.fProperty;
    fTrueName=dt.fTrueName;
  } return *this;
}

//______________________________________________________________________________
TDataType::~TDataType()
{
   // TDataType dtor deletes adopted G__TypedefInfo object.

   delete fInfo;
}

//______________________________________________________________________________
const char *TDataType::GetTypeName(EDataType type)
{
   // Return the name of the type.

   switch (type) {
      case  1: return "Char_t";
      case  2: return "Short_t";
      case  3: return "Int_t";
      case  4: return "Long_t";
      case  5: return "Float_t";
      case  6: return "Int_t";
      case  7: return "char*";
      case  8: return "Double_t";
      case  9: return "Double32_t";
      case 11: return "UChar_t";
      case 12: return "UShort_t";
      case 13: return "UInt_t";
      case 14: return "ULong_t";
      case 15: return "UInt_t";
      case 16: return "Long64_t";
      case 17: return "ULong64_t";
      case 18: return "Bool_t";
      case kOther_t:  return "";
      case kNoType_t: return "";
      case kchar:     return "Char_t";
   }
   return "";
}

//______________________________________________________________________________
const char *TDataType::GetTypeName() const
{
   // Get basic type of typedef, e,g.: "class TDirectory*" -> "TDirectory".
   // Result needs to be used or copied immediately.

   if (fInfo) {
      (const_cast<TDataType*>(this))->CheckInfo();
      return gInterpreter->TypeName(fTrueName.Data());
   } else {
      return fName.Data();
   }
}

//______________________________________________________________________________
const char *TDataType::GetFullTypeName() const
{
   // Get full type description of typedef, e,g.: "class TDirectory*".

   if (fInfo) {
      (const_cast<TDataType*>(this))->CheckInfo();
      return fTrueName;
   } else {
      return fName.Data();
   }
}

//______________________________________________________________________________
EDataType TDataType::GetType(const type_info &typeinfo)
{
   // Set type id depending on name.

   EDataType retType = kOther_t;

   if (!strcmp(typeid(unsigned int).name(), typeinfo.name())) {
      retType = kUInt_t;
   } else if (!strcmp(typeid(int).name(), typeinfo.name())) {
      retType = kInt_t;
   } else if (!strcmp(typeid(unsigned long).name(), typeinfo.name())) {
      retType = kULong_t;
   } else if (!strcmp(typeid(long).name(), typeinfo.name())) {
      retType = kLong_t;
   } else if (!strcmp(typeid(ULong64_t).name(), typeinfo.name())) {
      retType = kULong64_t;
   } else if (!strcmp(typeid(Long64_t).name(), typeinfo.name())) {
      retType = kLong64_t;
   } else if (!strcmp(typeid(unsigned short).name(), typeinfo.name())) {
      retType = kUShort_t;
   } else if (!strcmp(typeid(short).name(), typeinfo.name())) {
      retType = kShort_t;
   } else if (!strcmp(typeid(unsigned char).name(), typeinfo.name())) {
      retType = kUChar_t;
   } else if (!strcmp(typeid(char).name(), typeinfo.name())) {
      retType = kChar_t;
   } else if (!strcmp(typeid(bool).name(), typeinfo.name())) {
      retType = kBool_t;
   } else if (!strcmp(typeid(float).name(), typeinfo.name())) {
      retType = kFloat_t;
   } else if (!strcmp(typeid(double).name(), typeinfo.name())) {
      retType = kDouble_t;
   } else if (!strcmp(typeid(Double32_t).name(), typeinfo.name())) {
      retType = kDouble32_t;
   } else if (!strcmp(typeid(char*).name(), typeinfo.name())) {
      retType = kCharStar;
   }
   return retType;
}

//______________________________________________________________________________
const char *TDataType::AsString(void *buf) const
{
   // Return string containing value in buffer formatted according to
   // the basic data type. The result needs to be used or copied immediately.

   static char line[81];
   const char *name;

   if (fInfo) {
      (const_cast<TDataType*>(this))->CheckInfo();
      name = fTrueName;
   } else {
      name = fName.Data();
   }

   line[0] = 0;
   if (!strcmp("unsigned int", name))
      sprintf(line, "%u", *(unsigned int *)buf);
   else if (!strcmp("unsigned", name))
      sprintf(line, "%u", *(unsigned int *)buf);
   else if (!strcmp("int", name))
      sprintf(line, "%d", *(int *)buf);
   else if (!strcmp("unsigned long", name))
      sprintf(line, "%lu", *(unsigned long *)buf);
   else if (!strcmp("long", name))
      sprintf(line, "%ld", *(long *)buf);
   else if (!strcmp("unsigned long long", name))
      sprintf(line, "%llu", *(ULong64_t *)buf);
   else if (!strcmp("long long", name))
      sprintf(line, "%lld", *(Long64_t *)buf);
   else if (!strcmp("unsigned short", name))
      sprintf(line, "%hu", *(unsigned short *)buf);
   else if (!strcmp("short", name))
      sprintf(line, "%hd", *(short *)buf);
   else if (!strcmp("unsigned char", name))
      strncpy(line, (char *)buf,80);
   else if (!strcmp("bool", name))
      sprintf(line, "%s", *(bool *)buf ? "true" : "false");
   else if (!strcmp("char", name))
      strncpy(line, (char *)buf,80);
   else if (!strcmp("float", name))
      sprintf(line, "%g", *(float *)buf);
   else if (!strcmp("double", name))
      sprintf(line, "%g", *(double *)buf);
   else if (!strcmp("char*", name))
      sprintf(line, "%s", *(char**)buf);

   return line;
}

//______________________________________________________________________________
Long_t TDataType::Property() const
{
   // Get property description word. For meaning of bits see EProperty.

   if (fInfo) (const_cast<TDataType*>(this))->CheckInfo();
   return fProperty;
}

//______________________________________________________________________________
void TDataType::SetType(const char *name)
{
   // Set type id depending on name.

   fTrueName = name;
   fType = kOther_t;
   fSize = 0;

   if (!strcmp("unsigned int", name)) {
      fType = kUInt_t;
      fSize = sizeof(UInt_t);
   } else if (!strcmp("unsigned", name)) {
      fType = kUInt_t;
      fSize = sizeof(UInt_t);
   } else if (!strcmp("int", name)) {
      fType = kInt_t;
      fSize = sizeof(Int_t);
   } else if (!strcmp("unsigned long", name)) {
      fType = kULong_t;
      fSize = sizeof(ULong_t);
   } else if (!strcmp("long", name)) {
      fType = kLong_t;
      fSize = sizeof(Long_t);
   } else if (!strcmp("unsigned long long", name)) {
      fType = kULong64_t;
      fSize = sizeof(ULong64_t);
   } else if (!strcmp("long long", name)) {
      fType = kLong64_t;
      fSize = sizeof(Long64_t);
   } else if (!strcmp("unsigned short", name)) {
      fType = kUShort_t;
      fSize = sizeof(UShort_t);
   } else if (!strcmp("short", name)) {
      fType = kShort_t;
      fSize = sizeof(Short_t);
   } else if (!strcmp("unsigned char", name)) {
      fType = kUChar_t;
      fSize = sizeof(UChar_t);
   } else if (!strcmp("char", name)) {
      fType = kChar_t;
      fSize = sizeof(Char_t);
   } else if (!strcmp("bool", name)) {
      fType = kBool_t;
      fSize = sizeof(Bool_t);
   } else if (!strcmp("float", name)) {
      fType = kFloat_t;
      fSize = sizeof(Float_t);
   } else if (!strcmp("double", name)) {
      fType = kDouble_t;
      fSize = sizeof(Double_t);
   }

   if (!strcmp("Double32_t", fName.Data())) {
      fType = kDouble32_t;
   }
   if (!strcmp("char*",fName.Data())) {
      fType = kCharStar;
   }
   // kCounter =  6, kBits     = 15
}

//______________________________________________________________________________
Int_t TDataType::Size() const
{
   // Get size of basic typedef'ed type.

   if (fInfo) (const_cast<TDataType*>(this))->CheckInfo();
   return fSize;
}

//______________________________________________________________________________
void TDataType::CheckInfo()
{
   // Refresh the underlying information.

   // This can be needed if the library defining this typedef was loaded after
   // another library and that this other library is unloaded (in which case
   // things can get renumbered inside CINT).

   if (!fInfo) return;

   // This intentionally cast the constness away so that
   // we can call CheckInfo from const data members.

   if (!fInfo->IsValid() ||
       strcmp(fInfo->Name(),fName.Data())!=0) {

      // The fInfo is invalid or does not
      // point to this typedef anymore, let's
      // refresh it

      fInfo->Init(fName.Data());

      if (!fInfo->IsValid()) return;

      SetTitle(fInfo->Title());
      SetType(fInfo->TrueName());
      fProperty = fInfo->Property();
      fSize = fInfo->Size();
   }
}
