// @(#)root/meta:$Name:  $:$Id: TGenericClassInfo.cxx,v 1.2 2002/11/01 19:12:09 brun Exp $
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
        fVersion(1),
        fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0)
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
        fVersion(version),
        fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0)
   {
      Init(pragmabits);
   }

   TGenericClassInfo::TGenericClassInfo(const char *fullClassname, Int_t version,
                                        const char *declFileName, Int_t declFileLine,
                                        const type_info &info, const TInitBehavior  *action,
                                        VoidFuncPtr_t dictionary,
                                        IsAFunc_t isa, Int_t pragmabits)
      : fAction(action), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(isa), fShowMembers(0),
        fVersion(version),
        fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0)
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
         fClass->SetNew(fNew);
         fClass->SetNewArray(fNewArray);
         fClass->SetDelete(fDelete);
         fClass->SetDeleteArray(fDeleteArray);
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

   void TGenericClassInfo::SetNew(newFunc_t newFunc) 
   {
      fNew = newFunc;
      if (fClass) fClass->SetNew(fNew);
   }
   
   void TGenericClassInfo::SetNewArray(newArrFunc_t newArrayFunc)
   {
      fNewArray = newArrayFunc;
      if (fClass) fClass->SetNewArray(fNewArray);
   }
   
   void TGenericClassInfo::SetDelete(delFunc_t deleteFunc)
   {
      fDelete = deleteFunc;
      if (fClass) fClass->SetDelete(fDelete);
   }
   
   void TGenericClassInfo::SetDeleteArray(delArrFunc_t deleteArrayFunc)
   {
      fDeleteArray = deleteArrayFunc;
      if (fClass) fClass->SetDeleteArray(fDeleteArray);
   }
   
   void TGenericClassInfo::SetDestructor(desFunc_t destructorFunc)
   {
      fDestructor = destructorFunc;
      if (fClass) fClass->SetDestructor(fDestructor);
   }   

   newFunc_t TGenericClassInfo::GetNew() const
   {
      return fNew;
   }
 
   newArrFunc_t TGenericClassInfo::GetNewArray() const 
   {
      return fNewArray;
   }

   delFunc_t TGenericClassInfo::GetDelete() const
   {
      return fDelete;
   }

   delArrFunc_t TGenericClassInfo::GetDeleteArray() const
   {
      return fDeleteArray;
   }

   desFunc_t TGenericClassInfo::GetDestructor() const
   {
      return fDestructor;
   }

   
 

}
