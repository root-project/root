// @(#)root/meta:$Name:  $:$Id: TGenericClassInfo.cxx,v 1.15 2006/05/23 04:47:40 brun Exp $
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
#include "TVirtualIsAProxy.h"
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
                                        TVirtualIsAProxy *isa, Int_t pragmabits, Int_t sizof)
      : fAction(action), fClass(0), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info),
        fImplFileName(0), fImplFileLine(0),
        fIsA(isa), fShowMembers(showmembers),
        fVersion(1),
        fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fStreamer(0),
        fCollectionProxy(0), fSizeof(sizof)
   {
      // Constructor.

      Init(pragmabits);
   }

   TGenericClassInfo::TGenericClassInfo(const char *fullClassname, Int_t version,
                                        const char *declFileName, Int_t declFileLine,
                                        const type_info &info, const TInitBehavior  *action,
                                        void* showmembers,  VoidFuncPtr_t dictionary,
                                        TVirtualIsAProxy *isa, Int_t pragmabits, Int_t sizof)
      : fAction(action), fClass(0), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info),
        fImplFileName(0), fImplFileLine(0),
        fIsA(isa), fShowMembers(showmembers),
        fVersion(version),
        fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fStreamer(0),
        fCollectionProxy(0), fSizeof(sizof)
   {
      // Constructor with version number.

      Init(pragmabits);
   }

   TGenericClassInfo::TGenericClassInfo(const char *fullClassname, Int_t version,
                                        const char *declFileName, Int_t declFileLine,
                                        const type_info &info, const TInitBehavior  *action,
                                        VoidFuncPtr_t dictionary,
                                        TVirtualIsAProxy *isa, Int_t pragmabits, Int_t sizof)
      : fAction(action), fClass(0), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info),
        fImplFileName(0), fImplFileLine(0),
        fIsA(isa), fShowMembers(0),
        fVersion(version),
        fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fStreamer(0),
        fCollectionProxy(0), fSizeof(sizof)
   {
      // Constructor with version number and no showmembers.

      Init(pragmabits);
   }

   class TForNamespace {}; // Dummy class to give a typeid to namespace (See also TClassTable.cc)

   TGenericClassInfo::TGenericClassInfo(const char *fullClassname, Int_t version,
                                        const char *declFileName, Int_t declFileLine,
                                        const TInitBehavior  *action,
                                        VoidFuncPtr_t dictionary, Int_t pragmabits)
      : fAction(action), fClass(0), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(typeid(TForNamespace)),
        fImplFileName(0), fImplFileLine(0),
        fIsA(0), fShowMembers(0),
        fVersion(version),
        fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fStreamer(0),
        fCollectionProxy(0), fSizeof(0)
   {
      // Constructor for namespace

      Init(pragmabits);
   }

  /*  TGenericClassInfo::TGenericClassInfo(const TGenericClassInfo& gci) :
    fAction(gci.fAction),
    fClass(gci.fClass),
    fClassName(gci.fClassName),
    fDeclFileName(gci.fDeclFileName),
    fDeclFileLine(gci.fDeclFileLine),
    fDictionary(gci.fDictionary),
    fInfo(gci.fInfo),
    fImplFileName(gci.fImplFileName),
    fImplFileLine(gci.fImplFileLine),
    fIsA(gci.fIsA),
    fShowMembers(gci.fShowMembers),
    fVersion(gci.fVersion),
    fNew(gci.fNew),
    fNewArray(gci.fNewArray),
    fDelete(gci.fDelete),
    fDeleteArray(gci.fDeleteArray),
    fDestructor(gci.fDestructor),
    fStreamer(gci.fStreamer),
    fCollectionProxy(gci.fCollectionProxy),
    fSizeof(gci.fSizeof)
   { }

  TGenericClassInfo& TGenericClassInfo::operator=(const TGenericClassInfo& gci) 
   {
     if(this!=&gci) {
       fAction=gci.fAction;
       fClass=gci.fClass;
       fClassName=gci.fClassName;
       fDeclFileName=gci.fDeclFileName;
       fDeclFileLine=gci.fDeclFileLine;
       fDictionary=gci.fDictionary;
       fInfo=gci.fInfo;
       fImplFileName=gci.fImplFileName;
       fImplFileLine=gci.fImplFileLine;
       fIsA=gci.fIsA;
       fShowMembers=gci.fShowMembers;
       fVersion=gci.fVersion;
       fNew=gci.fNew;
       fNewArray=gci.fNewArray;
       fDelete=gci.fDelete;
       fDeleteArray=gci.fDeleteArray;
       fDestructor=gci.fDestructor;
       fStreamer=gci.fStreamer;
       fCollectionProxy=gci.fCollectionProxy;
       fSizeof=gci.fSizeof;
     } return *this;
   }
  */
   void TGenericClassInfo::Init(Int_t pragmabits)
   {
      // Initilization routine.

      if (!fAction) return;
      GetAction().Register(fClassName,
                           fVersion,
                           fInfo, // typeid(RootClass),
                           fDictionary,
                           pragmabits);
   }

   TGenericClassInfo::~TGenericClassInfo()
   {
      // Destructor.

      if (!fClass) delete fIsA; // fIsA is adopted by the class if any.
      fIsA = 0;
      if (!gROOT) return;
      if (fAction) GetAction().Unregister(GetClassName());
   }

   const TInitBehavior &TGenericClassInfo::GetAction() const
   {
      // Return the creator action.

      return *fAction;
   }

   TClass *TGenericClassInfo::GetClass()
   {
      // Generate and return the TClass object.

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
         fClass->SetDestructor(fDestructor);
         fClass->AdoptStreamer(fStreamer); fStreamer = 0;
         // If IsZombie is true, something went wront and we will not be
         // able to properly copy the collection proxy
         if (!fClass->IsZombie()
             && fCollectionProxy) fClass->CopyCollectionProxy(*fCollectionProxy);
         fClass->SetClassSize(fSizeof);
      }
      return fClass;
   }

   const char *TGenericClassInfo::GetClassName() const
   {
      // Return the class name

      return fClassName;
   }

   const type_info &TGenericClassInfo::GetInfo() const
   {
      // Return the typeifno value

      return fInfo;
   }

   void *TGenericClassInfo::GetShowMembers() const
   {
      // Return the point of the ShowMembers function
      return fShowMembers;
   }

   void TGenericClassInfo::SetFromTemplate()
   {
      // Import the information from the class template.

      TNamed *info = ROOT::RegisterClassTemplate(GetClassName(), 0, 0);
      if (info) SetImplFile(info->GetTitle(), info->GetUniqueID());
   }

   Int_t TGenericClassInfo::SetImplFile(const char *file, Int_t line)
   {
      // Set the name of the implementation file.

      fImplFileName = file;
      fImplFileLine = line;
      if (fClass) fClass->AddImplFile(file,line);
      return 0;
   }

   Int_t TGenericClassInfo::SetDeclFile(const char *file, Int_t line)
   {
      // Set the name of the declaration file.

      fDeclFileName = file;
      fDeclFileLine = line;
      if (fClass) fClass->SetDeclFile(file,line);
      return 0;
   }

   Short_t TGenericClassInfo::SetVersion(Short_t version)
   {
      // Set a class version number.

      ROOT::ResetClassVersion(fClass, GetClassName(),version);
      fVersion = version;
      return version;
   }

   Short_t TGenericClassInfo::AdoptStreamer(TClassStreamer *streamer)
   {
      // Set a Streamer object.  The streamer object is now 'owned'
      // by the TGenericClassInfo.

      delete fStreamer; fStreamer = 0;
      if (fClass) {
         fClass->AdoptStreamer(streamer);
      } else {
         fStreamer = streamer;
      }
      return 0;
   }

   Short_t TGenericClassInfo::AdoptCollectionProxy(TVirtualCollectionProxy *collProxy)
   {
      // Set the CollectProxy object.  The CollectionProxy object is now 'owned'
      // by the TGenericClassInfo.

      delete fCollectionProxy; fCollectionProxy = 0;
      fCollectionProxy = collProxy;
      if (fClass && fCollectionProxy && !fClass->IsZombie()) {
         fClass->CopyCollectionProxy(*fCollectionProxy);
      }
      return 0;
   }

   Short_t TGenericClassInfo::SetStreamer(ClassStreamerFunc_t streamer)
   {
      // Set a Streamer function.

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
      // Get the name of the declaring header file.

      return fDeclFileName;
   }

   Int_t TGenericClassInfo::GetDeclFileLine() const
   {
      // Get the declaring line number.

      return fDeclFileLine;
   }

   const char *TGenericClassInfo::GetImplFileName()
   {
      // Get the implementation filename.

      if (!fImplFileName) SetFromTemplate();
      return fImplFileName;
   }

   Int_t TGenericClassInfo::GetImplFileLine()
   {
      // Get the ClassImp line number.

      if (!fImplFileLine) SetFromTemplate();
      return fImplFileLine;
   }

   Int_t TGenericClassInfo::GetVersion() const
   {
      // Return the class version number.

      return fVersion;
   }

   TClass *TGenericClassInfo::IsA(const void *obj)
   {
      // Return the actual type of the object.

      return (*GetIsA())(obj);
   }

   TVirtualIsAProxy* TGenericClassInfo::GetIsA() const
   {
      // Return the IsA proxy.

      return fIsA;
   }

   void TGenericClassInfo::SetNew(NewFunc_t newFunc)
   {
      // Install a new wrapper around 'new'.

      fNew = newFunc;
      if (fClass) fClass->SetNew(fNew);
   }

   void TGenericClassInfo::SetNewArray(NewArrFunc_t newArrayFunc)
   {
      // Install a new wrapper around 'new []'.

      fNewArray = newArrayFunc;
      if (fClass) fClass->SetNewArray(fNewArray);
   }

   void TGenericClassInfo::SetDelete(DelFunc_t deleteFunc)
   {
      // Install a new wrapper around 'delete'.

      fDelete = deleteFunc;
      if (fClass) fClass->SetDelete(fDelete);
   }

   void TGenericClassInfo::SetDeleteArray(DelArrFunc_t deleteArrayFunc)
   {
      // Install a new wrapper around 'delete []'.

      fDeleteArray = deleteArrayFunc;
      if (fClass) fClass->SetDeleteArray(fDeleteArray);
   }

   void TGenericClassInfo::SetDestructor(DesFunc_t destructorFunc)
   {
      // Install a new wrapper around the destructor.

      fDestructor = destructorFunc;
      if (fClass) fClass->SetDestructor(fDestructor);
   }

   NewFunc_t TGenericClassInfo::GetNew() const
   {
      // Get the wrapper around 'new'.

      return fNew;
   }

   NewArrFunc_t TGenericClassInfo::GetNewArray() const
   {
      // Get the wrapper around 'new []'.

      return fNewArray;
   }

   DelFunc_t TGenericClassInfo::GetDelete() const
   {
      // Get the wrapper around 'delete'.

      return fDelete;
   }

   DelArrFunc_t TGenericClassInfo::GetDeleteArray() const
   {
      // Get the wrapper around 'delete []'.

      return fDeleteArray;
   }

   DesFunc_t TGenericClassInfo::GetDestructor() const
   {
      // Get the wrapper around the destructor.

      return fDestructor;
   }
}
