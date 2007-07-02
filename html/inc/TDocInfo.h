// @(#)root/html:$Name:  $:$Id: TDocInfo.h,v 1.3 2007/03/16 15:25:55 axel Exp $
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

#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TClassRef
#include "TClassRef.h"
#endif
#include <string>
#include <set>

class TClass;

class TModuleDocInfo;
//____________________________________________________________________
//
// Cache doc info for all known classes
//
class TClassDocInfo: public TObject {
public:
   // initialize the object
   TClassDocInfo(TClass* cl, const char* filename): 
      fClass(cl), fModule(0), fHtmlFileName(filename),
      fSelected(kTRUE), fHaveSource(kFALSE) { }
   virtual ~TClassDocInfo() {}

           TClass*         GetClass() const { return fClass; }
   virtual const char*     GetName() const;
           const char*     GetHtmlFileName() const { return fHtmlFileName; }

           void            SetModule(TModuleDocInfo* module) { fModule = module; }
           TModuleDocInfo* GetModule() const { return fModule; }

           void            SetSelected(Bool_t sel = kTRUE) { fSelected = sel; }
           Bool_t          IsSelected() const { return fSelected; }
           Bool_t          HaveSource() const { return fHaveSource; }
   
           void            SetHaveSource(Bool_t have = kTRUE) { fHaveSource = have; }

           ULong_t         Hash() const;

   virtual Bool_t          IsSortable() const { return kTRUE; }
   virtual Int_t           Compare(const TObject* obj) const;

private:
   TClassDocInfo();

   TClassRef               fClass; // class represented by this info object
   TModuleDocInfo*         fModule; // module this class is in
   TString                 fHtmlFileName; // name of the HTML doc file
   Bool_t                  fSelected; // selected for doc output
   Bool_t                  fHaveSource; // whether we can find the source locally

   ClassDef(TClassDocInfo,0); // info cache for class documentation
};

//____________________________________________________________________
//
// Cache doc info for all known modules
//
class TModuleDocInfo: public TNamed {
public:
   TModuleDocInfo(const char* name, const char* doc = ""): 
      TNamed(name, doc), fSelected(kTRUE) {}
   virtual ~TModuleDocInfo() {}

   void        SetDoc(const char* doc) { SetTitle(doc); }
   const char* GetDoc() const { return GetTitle(); }

   void        SetSelected(Bool_t sel = kTRUE) { fSelected = sel; }
   Bool_t      IsSelected() const { return fSelected; }

   void        AddClass(TClassDocInfo* cl) { fClasses.Add(cl); }
   TList*      GetClasses() { return &fClasses; }

   const TString& GetSourceDir() const { return fSourceDir; }
   void        SetSourceDir(const char* dir);

private:
   TList       fClasses;
   TString     fSourceDir; // (a) directory containing the modules' sources
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
