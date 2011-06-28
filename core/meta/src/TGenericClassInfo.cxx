// @(#)root/meta:$Id$
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
#include "TVirtualStreamerInfo.h"
#include "TStreamer.h"
#include "TVirtualIsAProxy.h"
#include "TVirtualCollectionProxy.h"
#include "TCollectionProxyInfo.h"
#include "TSchemaRule.h"
#include "TSchemaRuleSet.h"
#include "TError.h"
#include "TVirtualMutex.h"
#include "TInterpreter.h"

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
                                        ShowMembersFunc_t showmembers, VoidFuncPtr_t dictionary,
                                        TVirtualIsAProxy *isa, Int_t pragmabits, Int_t sizof)
      : fAction(action), fClass(0), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info),
        fImplFileName(0), fImplFileLine(0),
        fIsA(isa), fShowMembers(showmembers),
        fVersion(1),
        fMerge(0),fResetAfterMerge(0),fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fDirAutoAdd(0), fStreamer(0),
        fStreamerFunc(0), fCollectionProxy(0), fSizeof(sizof),
        fCollectionProxyInfo(0), fCollectionStreamerInfo(0)
   {
      // Constructor.

      Init(pragmabits);
   }

   TGenericClassInfo::TGenericClassInfo(const char *fullClassname, Int_t version,
                                        const char *declFileName, Int_t declFileLine,
                                        const type_info &info, const TInitBehavior  *action,
                                        ShowMembersFunc_t showmembers,  VoidFuncPtr_t dictionary,
                                        TVirtualIsAProxy *isa, Int_t pragmabits, Int_t sizof)
      : fAction(action), fClass(0), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info),
        fImplFileName(0), fImplFileLine(0),
        fIsA(isa), fShowMembers(showmembers),
        fVersion(version),
        fMerge(0),fResetAfterMerge(0),fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fDirAutoAdd(0), fStreamer(0),
        fStreamerFunc(0), fCollectionProxy(0), fSizeof(sizof),
        fCollectionProxyInfo(0), fCollectionStreamerInfo(0)
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
        fMerge(0),fResetAfterMerge(0),fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fDirAutoAdd(0), fStreamer(0),
        fStreamerFunc(0), fCollectionProxy(0), fSizeof(sizof),
        fCollectionProxyInfo(0), fCollectionStreamerInfo(0)

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
        fMerge(0),fResetAfterMerge(0),fNew(0),fNewArray(0),fDelete(0),fDeleteArray(0),fDestructor(0), fDirAutoAdd(0), fStreamer(0),
        fStreamerFunc(0), fCollectionProxy(0), fSizeof(0),
        fCollectionProxyInfo(0), fCollectionStreamerInfo(0)

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

      //TVirtualStreamerInfo::Class_Version MUST be the same as TStreamerInfo::Class_Version
      if (fVersion==-2) fVersion = TVirtualStreamerInfo::Class_Version();
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

      delete fCollectionProxyInfo;
      delete fCollectionStreamerInfo;
      delete fStreamer;
      if (!fClass) delete fIsA; // fIsA is adopted by the class if any.
      fIsA = 0;
      if (!gROOT || !gROOT->GetListOfClasses()) return;
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
         R__LOCKGUARD2(gCINTMutex);
         fClass = GetAction().CreateClass(GetClassName(),
                                          GetVersion(),
                                          GetInfo(),
                                          GetIsA(),
                                          GetShowMembers(),
                                          GetDeclFileName(),
                                          GetImplFileName(),
                                          GetDeclFileLine(),
                                          GetImplFileLine());
         fClass->SetNew(fNew);
         fClass->SetNewArray(fNewArray);
         fClass->SetDelete(fDelete);
         fClass->SetDeleteArray(fDeleteArray);
         fClass->SetDestructor(fDestructor);
         fClass->SetDirectoryAutoAdd(fDirAutoAdd);
         fClass->SetStreamerFunc(fStreamerFunc);
         fClass->SetMerge(fMerge);
         fClass->SetResetAfterMerge(fResetAfterMerge);
         fClass->AdoptStreamer(fStreamer); fStreamer = 0;
         // If IsZombie is true, something went wront and we will not be
         // able to properly copy the collection proxy
         if (!fClass->IsZombie()) {
            if (fCollectionProxy) fClass->CopyCollectionProxy(*fCollectionProxy);
            else if (fCollectionProxyInfo) {
               fClass->SetCollectionProxy(*fCollectionProxyInfo);
            }
         }
         fClass->SetClassSize(fSizeof);

         //---------------------------------------------------------------------
         // Attach the schema evolution information
         //---------------------------------------------------------------------
         CreateRuleSet( fReadRules, true );
         CreateRuleSet( fReadRawRules, false );
      }
      return fClass;
   }

   //---------------------------------------------------------------------------
   void TGenericClassInfo::CreateRuleSet( std::vector<TSchemaHelper>& vect,
                                          Bool_t ProcessReadRules )
   {
      // Attach the schema evolution information to TClassObject

      if ( vect.empty() ) {
         return;
      }

      //------------------------------------------------------------------------
      // Get the rules set
      //------------------------------------------------------------------------
      TSchemaRuleSet* rset = fClass->GetSchemaRules( kTRUE );

      //------------------------------------------------------------------------
      // Process the rules
      //------------------------------------------------------------------------
      TSchemaRule* rule;
      TString errmsg;
      std::vector<TSchemaHelper>::iterator it;
      for( it = vect.begin(); it != vect.end(); ++it ) {
         rule = new TSchemaRule();
         rule->SetTarget( it->fTarget );
         rule->SetTargetClass( fClass->GetName() );
         rule->SetSourceClass( it->fSourceClass );
         rule->SetSource( it->fSource );
         rule->SetCode( it->fCode );
         rule->SetVersion( it->fVersion );
         rule->SetChecksum( it->fChecksum );
         rule->SetEmbed( it->fEmbed );
         rule->SetInclude( it->fInclude );
         rule->SetAttributes( it->fAttributes );

         if( ProcessReadRules ) {
            rule->SetRuleType( TSchemaRule::kReadRule );
            rule->SetReadFunctionPointer( (TSchemaRule::ReadFuncPtr_t)it->fFunctionPtr );
         }
         else {
            rule->SetRuleType( TSchemaRule::kReadRawRule );
            rule->SetReadRawFunctionPointer( (TSchemaRule::ReadRawFuncPtr_t)it->fFunctionPtr );
         }
         if( !rset->AddRule( rule, TSchemaRuleSet::kCheckAll, &errmsg ) ) {
            ::Warning( "TGenericClassInfo", "The rule for class: \"%s\": version, \"%s\" and data members: \"%s\" has been skipped because %s.",
                        GetClassName(), it->fVersion.c_str(), it->fTarget.c_str(), errmsg.Data() );
            delete rule;
         }
      }
   }

   const char *TGenericClassInfo::GetClassName() const
   {
      // Return the class name

      return fClassName;
   }


   TCollectionProxyInfo *TGenericClassInfo::GetCollectionProxyInfo() const
   {
      // Return the set of info we have for the CollectionProxy, if any

      return fCollectionProxyInfo;
   }

   TCollectionProxyInfo *TGenericClassInfo::GetCollectionStreamerInfo() const
   {
      // Return the set of info we have for the Collection Streamer, if any

      return fCollectionProxyInfo;
   }

   const type_info &TGenericClassInfo::GetInfo() const
   {
      // Return the typeifno value

      return fInfo;
   }

   const std::vector<TSchemaHelper>& TGenericClassInfo::GetReadRawRules() const
   {
      // Return the list of rule give raw access to the TBuffer.

      return fReadRawRules;
   }


   const std::vector<TSchemaHelper>& TGenericClassInfo::GetReadRules() const
   {
      // Return the list of Data Model Evolution regular read rules.
      return fReadRules;
   }

   ShowMembersFunc_t TGenericClassInfo::GetShowMembers() const
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

   void TGenericClassInfo::AdoptCollectionProxyInfo(TCollectionProxyInfo *info)
   {
      // Set the info for the CollectionProxy and take ownership of the object
      // being passed

      delete fCollectionProxyInfo;;
      fCollectionProxyInfo = info;
   }

   void TGenericClassInfo::AdoptCollectionStreamerInfo(TCollectionProxyInfo *info)
   {
      // Set the info for the Collection Streamer and take ownership of the object
      // being passed

      delete fCollectionStreamerInfo;
      fCollectionStreamerInfo = info;
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

   void TGenericClassInfo::SetReadRawRules( const std::vector<TSchemaHelper>& rules )
   {
      // Set the list of Data Model Evolution read rules giving direct access to the TBuffer.
      fReadRawRules = rules;
   }


   void TGenericClassInfo::SetReadRules( const std::vector<TSchemaHelper>& rules )
   {
      // Set the list of Data Model Evolution regular read rules.
      fReadRules = rules;
   }

   Short_t TGenericClassInfo::SetStreamer(ClassStreamerFunc_t streamer)
   {
      // Set a External Streamer function.

      delete fStreamer; fStreamer = 0;
      if (fClass) {
         fClass->AdoptStreamer(new TClassStreamer(streamer));
      } else {
         fStreamer = new TClassStreamer(streamer);
      }
      return 0;
   }

   void TGenericClassInfo::SetStreamerFunc(ClassStreamerFunc_t streamer)
   {
      // Set a wrapper around the Streamer memger function.

      fStreamerFunc = streamer;
      if (fClass) fClass->SetStreamerFunc(streamer);
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

   void TGenericClassInfo::SetDirectoryAutoAdd(DirAutoAdd_t func)
   {
      // Install a new wrapper around SetDirectoryAutoAdd.

      fDirAutoAdd = func;
      if (fClass) fClass->SetDirectoryAutoAdd(fDirAutoAdd);
   }

   void TGenericClassInfo::SetMerge(MergeFunc_t func)
   {
      // Install a new wrapper around the Merge function.
      
      fMerge = func;
      if (fClass) fClass->SetMerge(fMerge);
   }
   
   void TGenericClassInfo::SetResetAfterMerge(ResetAfterMergeFunc_t func)
   {
      // Install a new wrapper around the Merge function.
      
      fResetAfterMerge = func;
      if (fClass) fClass->SetResetAfterMerge(fResetAfterMerge);
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
