// @(#)root/html:$Id$
// Author: Nenad Buncic   18/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDocInfo
#define ROOT_TDocInfo

#include "TClass.h"
#include "THashList.h"
#include "TNamed.h"
#include "TROOT.h"
#include <string>
#include <set>

class TDictionary;

class TModuleDocInfo;
//____________________________________________________________________
//
// Cache doc info for all known classes
//
class TClassDocInfo: public TObject {
public:
   // initialize the object
   TClassDocInfo(TClass* cl,
      const char* htmlfilename = "",
      const char* fsdecl = "", const char* fsimpl = "",
      const char* decl = 0, const char* impl = 0):
      fClass(cl), fModule(0), fHtmlFileName(htmlfilename),
      fDeclFileName(decl ? decl : cl->GetDeclFileName()),
      fImplFileName(impl ? impl : cl->GetImplFileName()),
      fDeclFileSysName(fsdecl), fImplFileSysName(fsimpl),
      fSelected(kTRUE) { }

   TClassDocInfo(TDictionary* cl,
      const char* htmlfilename = "",
      const char* fsdecl = "", const char* fsimpl = "",
      const char* decl = 0, const char* impl = 0):
      fClass(cl), fModule(0), fHtmlFileName(htmlfilename),
      fDeclFileName(decl),
      fImplFileName(impl),
      fDeclFileSysName(fsdecl), fImplFileSysName(fsimpl),
      fSelected(kTRUE) { }

   virtual ~TClassDocInfo()
   {
      // Required since we overload TObject::Hash.
      ROOT::CallRecursiveRemoveIfNeeded(*this);
   }

   TDictionary *GetClass() const { return fClass; }
   virtual const char*     GetName() const;
           const char*     GetHtmlFileName() const { return fHtmlFileName; }
           const char*     GetDeclFileName() const { return fDeclFileName; }
           const char*     GetImplFileName() const { return fImplFileName; }
           const char*     GetDeclFileSysName() const { return fDeclFileSysName; }
           const char*     GetImplFileSysName() const { return fImplFileSysName; }

           void            SetModule(TModuleDocInfo* module) { fModule = module; }
           TModuleDocInfo* GetModule() const { return fModule; }

           void            SetSelected(Bool_t sel = kTRUE) { fSelected = sel; }
           Bool_t          IsSelected() const { return fSelected; }
           Bool_t          HaveSource() const { return fDeclFileSysName.Length()
                                                   || (fClass && !dynamic_cast<TClass*>(fClass)); }

           void            SetHtmlFileName(const char* name) { fHtmlFileName = name; }
           void            SetDeclFileName(const char* name) { fDeclFileName = name; }
           void            SetImplFileName(const char* name) { fImplFileName = name; }
           void            SetDeclFileSysName(const char* fsname) { fDeclFileSysName = fsname; }
           void            SetImplFileSysName(const char* fsname) { fImplFileSysName = fsname; }

           ULong_t         Hash() const;

           TList&          GetListOfTypedefs() { return fTypedefs; }

   virtual Bool_t          IsSortable() const { return kTRUE; }
   virtual Int_t           Compare(const TObject* obj) const;

private:
   TClassDocInfo();

   TDictionary*            fClass; // class (or typedef) represented by this info object
   TModuleDocInfo*         fModule; // module this class is in
   TString                 fHtmlFileName; // name of the HTML doc file
   TString                 fDeclFileName; // header
   TString                 fImplFileName; // source
   TString                 fDeclFileSysName; // file system's location of the header
   TString                 fImplFileSysName; // file system's location of the source
   TList                   fTypedefs; // typedefs to this class
   Bool_t                  fSelected; // selected for doc output

   ClassDef(TClassDocInfo,0); // info cache for class documentation
};

//____________________________________________________________________
//
// Cache doc info for all known modules
//
class TModuleDocInfo: public TNamed {
public:
   TModuleDocInfo(const char* name, TModuleDocInfo* super, const char* doc = ""):
      TNamed(name, doc), fSuper(super), fSub(0), fSelected(kTRUE) {
         if (super) super->GetSub().Add(this);
      }
   virtual ~TModuleDocInfo() { fSub.Clear("nodelete"); fClasses.Clear("nodelete"); }

   void        SetDoc(const char* doc) { SetTitle(doc); }
   const char* GetDoc() const { return GetTitle(); }

   void        SetSelected(Bool_t sel = kTRUE) { fSelected = sel; }
   Bool_t      IsSelected() const { return fSelected; }

   void        AddClass(TClassDocInfo* cl) { fClasses.Add(cl); }
   TList*      GetClasses() { return &fClasses; }

   TModuleDocInfo* GetSuper() const { return fSuper; }
   THashList&  GetSub() { return fSub; }

private:
   TModuleDocInfo* fSuper; // module containing this module
   THashList   fSub; // modules contained in this module
   TList       fClasses;
   Bool_t      fSelected; // selected for doc output

   ClassDef(TModuleDocInfo,0); // documentation for a group of classes
};

//__________________________________________________________________________
//
// A library's documentation database:
// dependencies and sub-modules
//
class TLibraryDocInfo: public TNamed {
 public:
   TLibraryDocInfo() {}
   TLibraryDocInfo(const char* lib): TNamed(lib, "") {}

   std::set<std::string>& GetDependencies() {return fDependencies;}
   std::set<std::string>& GetModules() {return fModules;}
   void AddDependency(const std::string& lib) {fDependencies.insert(lib);}
   void AddModule(const std::string& module) {fModules.insert(module);}

 private:
   std::set<std::string> fDependencies; // dependencies on other libraries
   std::set<std::string> fModules; // modules in the library

   ClassDef(TLibraryDocInfo,0); // documentation for a library
};


#endif // ROOT_TDocInfo
