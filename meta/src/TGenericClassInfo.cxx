// @(#)root/meta:$Name:  $:$Id: TGenericClassInfo.cxx,v 1.3 2002/05/10 10:17:27 brun Exp $
// Author: Philippe Canal 08/05/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TClass.h"


namespace ROOT {

   const TInitBehavior *DefineBehavior(void * /*parent_type*/,
                                       void * /*actual_type*/)
   {

      // This function loads the default behavior for the
      // loading of classes.

      static TDefaultInitBehavior theDefault;
      return &theDefault;
   }

   TGenericClassInfo::TGenericClassInfo(const char *fullClassname,
                                        const char *declFileName, Int_t declFileLine,
                                        const type_info &info, const TInitBehavior  *action,
                                        void *showmembers, VoidFuncPtr_t dictionary,
                                        IsAFunc_t isa, Int_t pragmabits)
      : fAction(action), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(isa), fShowMembers(showmembers),
        fVersion(1)
   {
      Init(pragmabits);
   }
   
   TGenericClassInfo::TGenericClassInfo(const char *fullClassname, Int_t version,
                                        const char *declFileName, Int_t declFileLine,
                                        const type_info &info, const TInitBehavior  *action,
                                        void* showmembers,  VoidFuncPtr_t dictionary,
                                        IsAFunc_t isa, Int_t pragmabits)
      : fAction(action), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(isa), fShowMembers(showmembers),
        fVersion(version)
   {
      Init(pragmabits);
   }

   TGenericClassInfo::TGenericClassInfo(const char *fullClassname, Int_t version,
                                        const char *declFileName, Int_t declFileLine,
                                        const type_info &info, const TInitBehavior  *action,
                                        void* showmembers,  VoidFuncPtr_t dictionary,
                                        Int_t pragmabits)
      : fAction(action), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(0), fShowMembers(showmembers),
        fVersion(version)
   {
      Init(pragmabits);
   }

   TGenericClassInfo::TGenericClassInfo(const char *fullClassname, Int_t version,
                                        const char *declFileName, Int_t declFileLine,
                                        const type_info &info, const TInitBehavior  *action,
                                        VoidFuncPtr_t dictionary,
                                        Int_t pragmabits)
      : fAction(action), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(0), fShowMembers(0),
        fVersion(version)
   {
      Init(pragmabits);
   }

   void TGenericClassInfo::Init(Int_t pragmabits)
   {
      if (!fAction) return;
      GetAction().Register(fClassName,
                           fVersion,
                           fInfo, // typeid(RootClass),
                           fDictionary,
                           pragmabits);
   }

   TGenericClassInfo::~TGenericClassInfo()
   {
      if (!gROOT) return;
      if (fAction) GetAction().Unregister(GetClassName());
   }

   const TInitBehavior &TGenericClassInfo::GetAction() const
   {
      return *fAction;
   }

   TClass *TGenericClassInfo::GetClass()
   {
      if (!fClass && fAction) {
         fClass = GetAction().CreateClass(GetClassName(),
                                          GetVersion(),
                                          GetInfo(),
                                          GetIsA(),
                                          (ShowMembersFunc_t)GetShowMembers(),
                                          GetDeclFileName(),
                                          GetImplFileName(),
                                          GetDeclFileLine(),
                                          GetImplFileLine());
      }
      return fClass;
   }

   const char *TGenericClassInfo::GetClassName() const
   {
      return fClassName;
   }

   const type_info &TGenericClassInfo::GetInfo() const
   {
      return fInfo;
   }

   void *TGenericClassInfo::GetShowMembers() const
   {
      return fShowMembers;
   }

   void TGenericClassInfo::SetFromTemplate()
   {
      TNamed *info = ROOT::RegisterClassTemplate(GetClassName(), 0, 0);
      if (info) SetImplFile(info->GetTitle(), info->GetUniqueID());
   }

   Int_t TGenericClassInfo::SetImplFile(const char *file, Int_t line)
   {
      fImplFileName = file;
      fImplFileLine = line;
      if (fClass) fClass->AddImplFile(file,line);
      return 0;
   }

   Short_t TGenericClassInfo::SetVersion(Short_t version)
   {
      ROOT::ResetClassVersion(fClass, GetClassName(),version);
      fVersion = version;
      return version;
   }

   const char *TGenericClassInfo::GetDeclFileName() const
   {
      return fDeclFileName;
   }

   Int_t TGenericClassInfo::GetDeclFileLine() const
   {
      return fDeclFileLine;
   }

   const char *TGenericClassInfo::GetImplFileName()
   {
      if (!fImplFileName) SetFromTemplate();
      return fImplFileName;
   }

   Int_t TGenericClassInfo::GetImplFileLine()
   {
      if (!fImplFileLine) SetFromTemplate();
      return fImplFileLine;
   }

   Int_t TGenericClassInfo::GetVersion() const
   {
      return fVersion;
   }

   TClass *TGenericClassInfo::IsA(const void *obj)
   {
      return (GetIsA())(obj);
   }

   IsAFunc_t TGenericClassInfo::GetIsA() const
   {
      return fIsA;
   }

}
