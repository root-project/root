// @(#)root/meta:$Name:  $:$Id: TGenericClassInfo.cxx,v 1.5 2004/01/10 10:52:30 brun Exp $
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
#include "TStreamer.h"
#include "TVirtualCollectionProxy.h"

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
                                        IsAFunc_t isa, Int_t pragmabits, Int_t sizof)
      : fAction(action), fClass(0), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(isa), fShowMembers(showmembers),
        fVersion(1), 
        fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fStreamer(0),
        fCollectionProxy(0), fSizeof(sizof)
   {
      Init(pragmabits);
   }
   
   TGenericClassInfo::TGenericClassInfo(const char *fullClassname, Int_t version,
                                        const char *declFileName, Int_t declFileLine,
                                        const type_info &info, const TInitBehavior  *action,
                                        void* showmembers,  VoidFuncPtr_t dictionary,
                                        IsAFunc_t isa, Int_t pragmabits, Int_t sizof)
      : fAction(action), fClass(0), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(isa), fShowMembers(showmembers),
        fVersion(version),
        fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fStreamer(0), 
        fCollectionProxy(0), fSizeof(sizof)
   {
      Init(pragmabits);
   }

   TGenericClassInfo::TGenericClassInfo(const char *fullClassname, Int_t version,
                                        const char *declFileName, Int_t declFileLine,
                                        const type_info &info, const TInitBehavior  *action,
                                        VoidFuncPtr_t dictionary,
                                        IsAFunc_t isa, Int_t pragmabits, Int_t sizof)
      : fAction(action), fClass(0), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(isa), fShowMembers(0),
        fVersion(version),
        fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fStreamer(0), 
        fCollectionProxy(0), fSizeof(sizof)
   {
      Init(pragmabits);
   }

   class fornamespace {}; // Dummy class to give a typeid to namespace 

   TGenericClassInfo::TGenericClassInfo(const char *fullClassname, Int_t version,
                                        const char *declFileName, Int_t declFileLine,
                                        const TInitBehavior  *action,
                                        VoidFuncPtr_t dictionary, Int_t pragmabits)
      : fAction(action), fClass(0), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(typeid(fornamespace)), fIsA(0), fShowMembers(0),
        fVersion(version),
        fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fStreamer(0), 
        fCollectionProxy(0), fSizeof(0)
   {
      // Constructor for namespace

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
         fClass->AdoptStreamer(fStreamer); fStreamer = 0;
         if (fCollectionProxy) fClass->CopyCollectionProxy(*fCollectionProxy);
         fClass->SetClassSize(fSizeof);
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

   Short_t TGenericClassInfo::AdoptStreamer(TClassStreamer *streamer) {
      delete fStreamer; fStreamer = 0;
      if (fClass) {
         fClass->AdoptStreamer(streamer);
      } else {
         fStreamer = streamer;
      }
      return 0;
   }

   Short_t TGenericClassInfo::AdoptCollectionProxy(TVirtualCollectionProxy *collProxy) {
      delete fCollectionProxy; fCollectionProxy = 0;
      fCollectionProxy = collProxy;
      if (fClass && fCollectionProxy) fClass->CopyCollectionProxy(*fCollectionProxy);
      return 0;
   }

   Short_t TGenericClassInfo::SetStreamer(ClassStreamerFunc_t streamer) {
      delete fStreamer; fStreamer = 0;
      if (fClass) {
         fClass->AdoptStreamer(new TClassStreamer(streamer));
      } else {
         fStreamer = new TClassStreamer(streamer);
      }
      return 0;
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

   void TGenericClassInfo::SetNew(NewFunc_t newFunc) 
   {
      fNew = newFunc;
      if (fClass) fClass->SetNew(fNew);
   }
   
   void TGenericClassInfo::SetNewArray(NewArrFunc_t newArrayFunc)
   {
      fNewArray = newArrayFunc;
      if (fClass) fClass->SetNewArray(fNewArray);
   }
   
   void TGenericClassInfo::SetDelete(DelFunc_t deleteFunc)
   {
      fDelete = deleteFunc;
      if (fClass) fClass->SetDelete(fDelete);
   }
   
   void TGenericClassInfo::SetDeleteArray(DelArrFunc_t deleteArrayFunc)
   {
      fDeleteArray = deleteArrayFunc;
      if (fClass) fClass->SetDeleteArray(fDeleteArray);
   }
   
   void TGenericClassInfo::SetDestructor(DesFunc_t destructorFunc)
   {
      fDestructor = destructorFunc;
      if (fClass) fClass->SetDestructor(fDestructor);
   }   

   NewFunc_t TGenericClassInfo::GetNew() const
   {
      return fNew;
   }
 
   NewArrFunc_t TGenericClassInfo::GetNewArray() const 
   {
      return fNewArray;
   }

   DelFunc_t TGenericClassInfo::GetDelete() const
   {
      return fDelete;
   }

   DelArrFunc_t TGenericClassInfo::GetDeleteArray() const
   {
      return fDeleteArray;
   }

   DesFunc_t TGenericClassInfo::GetDestructor() const
   {
      return fDestructor;
   }
   
 

}
