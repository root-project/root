// @(#)root/meta:$Name:  $:$Id: $
// Author: Philippe Canal 08/05/2002

#include "Rtypes.h"
#include "TNamed.h"
#include "TClass.h"



namespace ROOT {
   
   const InitBehavior *DefineBehavior(void * /*parent_type*/,
                                      void * /*actual_type*/) {

      // This function loads the default behavior for the 
      // loading of classes.

      static DefaultInitBehavior Default;
      return &Default;
   }

   GenericClassInfo::GenericClassInfo(const char *fullClassname, 
                                      const char *declFileName, Int_t declFileLine,
                                      const type_info &info, const InitBehavior  *action,
                                      void *showmembers, VoidFuncPtr_t dictionary, 
                                      IsAFunc_t isa, Int_t pragmabits) 
      : fAction(action), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(isa), fShowMembers(showmembers), 
        fVersion(1)
   {
      Init(pragmabits);
   }
   
   GenericClassInfo::GenericClassInfo(const char *fullClassname, Int_t version,
                                      const char *declFileName, Int_t declFileLine,
                                      const type_info &info, const InitBehavior  *action,
                                      void* showmembers,  VoidFuncPtr_t dictionary, 
                                      IsAFunc_t isa, Int_t pragmabits)
      : fAction(action), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(isa), fShowMembers(showmembers), 
        fVersion(version) {
      Init(pragmabits);
   }
   
   GenericClassInfo::GenericClassInfo(const char *fullClassname, Int_t version,
                                      const char *declFileName, Int_t declFileLine,
                                      const type_info &info, const InitBehavior  *action,
                                      void* showmembers,  VoidFuncPtr_t dictionary, 
                                      Int_t pragmabits)
      : fAction(action), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(0), fShowMembers(showmembers), 
        fVersion(version) {
      Init(pragmabits);
   }
 
   GenericClassInfo::GenericClassInfo(const char *fullClassname, Int_t version,
                                      const char *declFileName, Int_t declFileLine,
                                      const type_info &info, const InitBehavior  *action,
                                      VoidFuncPtr_t dictionary, 
                                      Int_t pragmabits)
      : fAction(action), fClassName(fullClassname),
        fDeclFileName(declFileName), fDeclFileLine(declFileLine),
        fDictionary(dictionary), fInfo(info), fIsA(0), fShowMembers(0), 
        fVersion(version) {
      Init(pragmabits);
   }
   
   void GenericClassInfo::Init(Int_t pragmabits) {
      if (!fAction) return;
      GetAction().Register(fClassName,
                           fVersion,
                           fInfo, // typeid(RootClass),
                           fDictionary,
                           pragmabits);
   }
   
   GenericClassInfo::~GenericClassInfo() { if (fAction) GetAction().Unregister(GetClassName()); }
   
   const InitBehavior &GenericClassInfo::GetAction() {
      //if (!fAction) {
      //   RootClass *ptr = 0;
      //   fAction = DefineBehavior(ptr, ptr);
      //}
      return *fAction;
   }
      
   TClass *GenericClassInfo::GetClass() {
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
   
   const char *GenericClassInfo::GetClassName() {
      return fClassName;
   }
   
   const type_info &GenericClassInfo::GetInfo() {
      return fInfo;
   }
   
   void *GenericClassInfo::GetShowMembers() {
      return fShowMembers;
   }
   
   void GenericClassInfo::SetFromTemplate() {
      TNamed *info = ROOT::RegisterClassTemplate(GetClassName(), 0, 0);
      if (info) SetImplFile(info->GetTitle(), info->GetUniqueID());
   }

   int GenericClassInfo::SetImplFile(const char *file, Int_t line) {
      fImplFileName = file;
      fImplFileLine = line;
      if (fClass) fClass->AddImplFile(file,line);
      return 0;
   }

   Short_t GenericClassInfo::SetVersion(Short_t version) {
      ROOT::ResetClassVersion(fClass, GetClassName(),version);
      fVersion = version;
      return version;
   }
   
   const char *GenericClassInfo::GetDeclFileName() {
      return fDeclFileName;
   }
   
   Int_t GenericClassInfo::GetDeclFileLine() {
      return fDeclFileLine;
   }
   
   const char *GenericClassInfo::GetImplFileName() {
      if (!fImplFileName) SetFromTemplate();
      return fImplFileName;
   }
   
   Int_t GenericClassInfo::GetImplFileLine() {
      if (!fImplFileLine) SetFromTemplate(); 
      return fImplFileLine;
   }
   
   Int_t GenericClassInfo::GetVersion() {
      return fVersion;
   }
   
   TClass* GenericClassInfo::IsA(const void *obj) {
      return (GetIsA())(obj);
   }
   
   IsAFunc_t GenericClassInfo::GetIsA() { return fIsA; }

}
