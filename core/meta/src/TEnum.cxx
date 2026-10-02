// @(#)root/meta:$Id$
// Author: Bianca-Cristina Cristescu   10/07/13

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TEnum
The TEnum class implements the enum type.
*/

#include <iostream>

#include "TEnum.h"
#include "TEnumConstant.h"
#include "TInterpreter.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClassTable.h"
#include "TProtoClass.h"
#include "TROOT.h"

#include "TListOfEnums.h"


////////////////////////////////////////////////////////////////////////////////
/// Constructor for TEnum class.
/// It takes the name of the TEnum type, interpreter info and surrounding class
/// the enum it is not globalat namespace scope.
/// Constant List is owner if enum not on global scope (thus constants not
/// in TROOT::GetListOfGlobals).

TEnum::TEnum(const char *name, DeclId_t declid, TClass *cls)
   : fClass(cls)
{
   SetName(name);
   if (cls) {
      fConstantList.SetOwner(kTRUE);
   }

   // Determine fQualName
   if (0 != strcmp("",GetTitle())){ // It comes from a protoclass
      fQualName = std::string(GetTitle()) + "::" + GetName();
   }
   else if (GetClass()){ // It comes from a class/ns
      fQualName = std::string(GetClass()->GetName()) + "::" + GetName();
   }
   else { // it is in the global scope
      fQualName = GetName();
   }

   Update(declid);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TEnum::TEnum(const TEnum &src) : TDictionary(src)
{
   fClass = src.fClass;
   fInfo = src.fInfo ? gInterpreter->ClassInfo_Factory(src.fInfo) : nullptr;
   fQualName = src.fQualName;
   fUnderlyingType = src.fUnderlyingType;

   Bool_t isowner = src.fConstantList.IsOwner();
   fConstantList.SetOwner(isowner);
   TIter next(&src.fConstantList);
   while (auto c = (TEnumConstant *) next())
      fConstantList.Add(isowner ? new TEnumConstant(*c) : c);

}

////////////////////////////////////////////////////////////////////////////////
/// Assign operator

TEnum& TEnum::operator=(const TEnum &src)
{
   if (this != &src) {
      if (fInfo)
         gInterpreter->ClassInfo_Delete(fInfo);
      fConstantList.Clear();

      TDictionary::operator=(src);

      fInfo = src.fInfo ? gInterpreter->ClassInfo_Factory(src.fInfo) : nullptr;
      fQualName = src.fQualName;
      fUnderlyingType = src.fUnderlyingType;

      Bool_t isowner = src.fConstantList.IsOwner();
      fConstantList.SetOwner(isowner);
      TIter next(&src.fConstantList);
      while (auto c = (TEnumConstant *) next())
         fConstantList.Add(isowner ? new TEnumConstant(*c) : c);
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TEnum::~TEnum()
{
   gInterpreter->ClassInfo_Delete(fInfo);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a EnumConstant to the list of constants of the Enum Type.

void TEnum::AddConstant(TEnumConstant *constant)
{
   fConstantList.Add(constant);
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if this enum object is pointing to a currently
/// loaded enum.  If a enum is unloaded after the TEnum
/// is created, the TEnum will be set to be invalid.

Bool_t TEnum::IsValid()
{
   if (TestBit(kBitIsValid))
      return true;

   // Register the transaction when checking the validity of the object.
   if (!fInfo && UpdateInterpreterStateMarker()) {
      DeclId_t newId = gInterpreter->GetEnum(fClass, fName);
      if (newId)
         Update(newId);
      return newId != nullptr;
   }
   return fInfo != nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get property description word. For meaning of bits see EProperty.

Long_t TEnum::Property() const
{
   return kIsEnum | (TestBit(kBitIsScopedEnum) ? kIsScopedEnum : 0);
}

////////////////////////////////////////////////////////////////////////////////

TDictionary::DeclId_t TEnum::GetDeclId() const
{
   if (fInfo)
      return gInterpreter->GetDeclId(fInfo);

   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

void TEnum::Update(DeclId_t id)
{
   if (fInfo)
      gInterpreter->ClassInfo_Delete(fInfo);
   if (!id) {
      ResetBit(kBitIsValid);
      fInfo = nullptr;
      return;
   }

   fInfo = gInterpreter->ClassInfo_Factory(id);

   if (fInfo) {
      SetBit(kBitIsScopedEnum, gInterpreter->ClassInfo_IsScopedEnum(fInfo));
      fUnderlyingType = gInterpreter->ClassInfo_GetUnderlyingType(fInfo);
      SetBit(kBitIsValid);
   } else {
      ResetBit(kBitIsValid);
   }
}

////////////////////////////////////////////////////////////////////////////////

TEnum *TEnum::GetEnum(const std::type_info &ti, ESearchAction sa)
{
   int errorCode = 0;
   char *demangledEnumName = TClassEdit::DemangleName(ti.name(), errorCode);

   if (errorCode != 0) {
      free(demangledEnumName);
      std::cerr << "ERROR TEnum::GetEnum - A problem occurred while demangling name.\n";
      return nullptr;
   }

   const char *constDemangledEnumName = demangledEnumName;
   TEnum *en = TEnum::GetEnum(constDemangledEnumName, sa);
   free(demangledEnumName);
   return en;

}

////////////////////////////////////////////////////////////////////////////////
/// Static function to retrieve enumerator from the ROOT's typesystem.
/// The search is carried out in up to three passes:
/// 1. with the name as given, restricted to the enums already registered in
///    the typesystem lists (no autoloading, no interpreter lookup);
/// 2. on a miss, with the name normalized to resolve typedefs and using
///    declarations (see #15406); the normalization is computed with both
///    autoloading and autoparsing suspended, so that it cannot load
///    libraries or parse dictionary payloads as a side effect (see #18923),
///    and the search then runs at the requested search level;
/// 3. if the requested search level allows interpreter lookups
///    (kInterpLookup), the normalization is repeated with autoparsing
///    allowed - resolving names whose typedefs require parsing a header
///    first - and the search is repeated if this yields a different name.
/// In each pass there are two top level code paths: the enumerator is scoped
/// or isn't. If it is not, a lookup in the list of global enums is performed.
/// If it is, two lookups are carried out for its scope: one in the list of
/// classes and one in the list of protoclasses. If a scope with the desired name
/// is found, the enum is searched. If the scope is not found, and the load flag is
/// true, the aforementioned two steps are performed again after an autoload attempt
/// with the name of the scope as key is tried out.
/// If the interpreter lookup flag is false, the ListOfEnums objects are not treated
/// as such, but rather as THashList objects. This prevents any flow of information
/// from the interpreter into the ROOT's typesystem: a snapshot of the typesystem
/// status is taken.

TEnum *TEnum::GetEnum(const char *enumName, ESearchAction sa)
{
   // Potential optimisation: reduce number of branches using partial specialisation of
   // helper functions.

   // Wrap some gymnastic around the enum finding. The special treatment of the
   // ListOfEnums objects is located in this routine.
   auto findEnumInList = [](const TCollection * l, const char * enName, ESearchAction sa_local) {
      TObject *obj;
      if (sa_local & kInterpLookup) {
         obj = l->FindObject(enName);
      } else {
         auto enumTable = dynamic_cast<const TListOfEnums *>(l);
         obj = enumTable->GetObject(enName);
      }
      return static_cast<TEnum *>(obj);
   };

   // Helper routine to look fo the scope::enum in the typesystem.
   // If autoload and interpreter lookup is allowed, TClass::GetClass is called.
   // If not, the list of classes and the list of protoclasses is inspected.
   auto searchEnum = [findEnumInList](const char *scopeName, const char *enName, ESearchAction sa_local) -> TEnum * {
      // Check if the scope is a class
      if (sa_local == (kALoadAndInterpLookup)) {
         auto scope = TClass::GetClass(scopeName, true);
         TEnum *en = nullptr;
         if (scope)
            en = findEnumInList(scope->GetListOfEnums(kFALSE), enName, sa_local);
         return en;
      }

      if (auto tClassScope = static_cast<TClass *>(gROOT->GetListOfClasses()->FindObject(scopeName))) {
         // If this is a class, load only if the user allowed interpreter lookup
         // If this is a namespace and the user did not allow for interpreter lookup, load but before disable
         // autoparsing if enabled.
         bool canLoadEnums (sa_local & kInterpLookup);
         const bool scopeIsNamespace (tClassScope->Property() & kIsNamespace);

         const bool autoParseSuspended = gInterpreter->IsAutoParsingSuspended();

         if (scopeIsNamespace && !autoParseSuspended) {
            // Lock down the autoparsing state.
            R__WRITE_LOCKGUARD(ROOT::gCoreMutex);
            TInterpreter::SuspendAutoParsing autoParseRaii(gInterpreter, true);

            auto listOfEnums = tClassScope->GetListOfEnums(true);
            // Previous incarnation of the code re-enabled the auto parsing,
            // before executing findEnumInList
            return findEnumInList(listOfEnums, enName, sa_local);
         } else {
            auto listOfEnums = tClassScope->GetListOfEnums(canLoadEnums);
            return findEnumInList(listOfEnums, enName, sa_local);
         }
      }
      // Check if the scope is still a protoclass
      else if (auto tProtoClassscope = static_cast<TProtoClass *>((gClassTable->GetProtoNorm(scopeName)))) {
         auto listOfEnums = tProtoClassscope->GetListOfEnums();
         if (listOfEnums)
            return findEnumInList(listOfEnums, enName, sa_local);
      }
      return nullptr;
   };

   // Run the search with the given name and search level. skipListLookup
   // skips the initial list-only (kNone-level) lookup, for callers that
   // already know it missed for this name. Returns nullptr if no enum was
   // found.
   auto runSearch = [&searchEnum, &findEnumInList](const char *name, ESearchAction sa_arg,
                                                   bool skipListLookup) -> TEnum * {
      TEnum *theEnum = nullptr;
      const char *lastPos = TClassEdit::GetUnqualifiedName(name);

      // Keep the state consistent.  In particular prevent change in the state of
      // AutoLoading and AutoParsing allowance and gROOT->GetListOfClasses()
      // and the later update/modification to the autoparsing state.
      R__READ_LOCKGUARD(ROOT::gCoreMutex);

      if (lastPos != name) {
         // We have a scope
         const auto enName = lastPos;
         std::string scopeName{name, static_cast<std::size_t>(lastPos - name) - 2};
         // Three levels of search
         if (!skipListLookup)
            theEnum = searchEnum(scopeName.c_str(), enName, kNone);
         if (!theEnum && (sa_arg & kAutoload)) {
            const auto libsLoaded = gInterpreter->AutoLoad(scopeName.c_str());
            // It could be an enum in a scope which is not selected
            if (libsLoaded == 0) {
               gInterpreter->AutoLoad(name);
            }
            theEnum = searchEnum(scopeName.c_str(), enName, kAutoload);
         }
         if (!theEnum && (sa_arg & kInterpLookup)) {
            if (gDebug > 0) {
               printf("TEnum::GetEnum: Header Parsing - The enumerator %s is not known to the typesystem: an "
                      "interpreter lookup will be performed. This can imply parsing of headers. This can be avoided "
                      "selecting the numerator in the linkdef/selection file.\n",
                      name);
            }
            theEnum = searchEnum(scopeName.c_str(), enName, kALoadAndInterpLookup);
         }
      } else {
         // We don't have any scope: this is a global enum
         if (!skipListLookup)
            theEnum = findEnumInList(gROOT->GetListOfEnums(), name, kNone);
         if (!theEnum && (sa_arg & kAutoload)) {
            gInterpreter->AutoLoad(name);
            theEnum = findEnumInList(gROOT->GetListOfEnums(), name, kAutoload);
         }
         if (!theEnum && (sa_arg & kInterpLookup)) {
            if (gDebug > 0) {
               printf("TEnum::GetEnum: Header Parsing - The enumerator %s is not known to the typesystem: an "
                      "interpreter lookup will be performed. This can imply parsing of headers. This can be avoided "
                      "selecting the numerator in the linkdef/selection file.\n",
                      name);
            }
            theEnum = findEnumInList(gROOT->GetListOfEnums(), name, kALoadAndInterpLookup);
         }
      }

      return theEnum;
   };

   const char *lastPos = TClassEdit::GetUnqualifiedName(enumName);

   if (strchr(lastPos,'<')) {
      // The unqualified name has template syntax, it can't possibly be an
      // enum.
      return nullptr;
   }

   // Pass 1: search with the name as given, restricted to what is already
   // registered in ROOT's typesystem lists (kNone: no autoloading and no
   // header parsing). This avoids interpreter round trips (and the
   // autoloading and autoparsing they can trigger) for names that are
   // already known, notably while dictionaries are being registered (see
   // #18923).
   if (TEnum *en = runSearch(enumName, kNone, false))
      return en;

   // Pass 2: the enum is not already known under the name as given;
   // normalize the name to resolve typedefs and using declarations (see
   // #15406) and run the search at the requested level.
   // The normalization may call into the interpreter; suspend both
   // AutoLoading and AutoParsing to keep it free of those side effects (a
   // dictionary payload parsed halfway through a dictionary registration
   // can leave the interpreter in an inconsistent state, see #18923).
   std::string normalizedName;
   {
      R__WRITE_LOCKGUARD(ROOT::gCoreMutex);
      TInterpreter::SuspendAutoLoadingRAII autoloadOff(gInterpreter);
      TInterpreter::SuspendAutoParsing autoParseRaii(gInterpreter, true);
      TClassEdit::GetNormalizedName(normalizedName, enumName);
   }

   const bool nameChanged = normalizedName != enumName;
   if (nameChanged || sa != kNone) {
      if (TEnum *en = runSearch(nameChanged ? normalizedName.c_str() : enumName, sa,
                                /*skipListLookup=*/!nameChanged))
         return en;
   }

   // Pass 3: the caller explicitly allows interpreter lookups that can imply
   // parsing headers; repeat the normalization with autoparsing allowed (as
   // before this restructuring), to also resolve names whose typedefs or
   // using declarations only become visible by parsing a header, and search
   // again if this yields a different name than the passes above.
   if (sa & kInterpLookup) {
      std::string reparsedName;
      {
         R__WRITE_LOCKGUARD(ROOT::gCoreMutex);
         TInterpreter::SuspendAutoLoadingRAII autoloadOff(gInterpreter);
         TClassEdit::GetNormalizedName(reparsedName, enumName);
      }
      if (reparsedName != normalizedName)
         return runSearch(reparsedName.c_str(), sa, false);
   }

   return nullptr;
}
