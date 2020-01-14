/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <ROOT/RBrowsableTDirectory.hxx>

#include <ROOT/Browsable/RLevelIter.hxx>
#include <ROOT/Browsable/RProvider.hxx>

#include <ROOT/RBrowsableTObject.hxx>

#include "ROOT/RLogger.hxx"

#include "TSystem.h"
#include "TKey.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"

using namespace std::string_literals;

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Browsable;


/** \class TDirectoryLevelIter
\ingroup rbrowser

Iterator over keys in TDirectory
*/


class TDirectoryLevelIter : public RLevelIter {
   TDirectory *fDir{nullptr};         ///<! current directory handle
   std::unique_ptr<TIterator>  fIter; ///<! created iterator
   TKey *fKey{nullptr};               ///<! currently selected key
   std::string fCurrentName;          ///<! current key name

   bool CreateIter()
   {
      if (!fDir) return false;
      fIter.reset(fDir->GetListOfKeys()->MakeIterator());
      fKey = nullptr;
      return true;
   }

   bool NextDirEntry()
   {
      fCurrentName.clear();
      if (!fIter) return false;

      fKey = dynamic_cast<TKey *>(fIter->Next());

      if (!fKey) {
         fIter.reset();
         return false;
      }

      fCurrentName = fKey->GetName();
      fCurrentName.append(";");
      fCurrentName.append(std::to_string(fKey->GetCycle()));

      return true;
   }

public:
   explicit TDirectoryLevelIter(TDirectory *dir) : fDir(dir) { CreateIter(); }

   virtual ~TDirectoryLevelIter() = default;

   bool Reset() override { return CreateIter(); }

   bool Next() override { return NextDirEntry(); }

   // use default implementation for now
   // bool Find(const std::string &name) override { return FindDirEntry(name); }

   bool HasItem() const override { return fKey && !fCurrentName.empty(); }

   std::string GetName() const override { return fCurrentName; }

   int CanHaveChilds() const override
   {
      std::string clname = fKey->GetClassName();
      return (clname.find("TDirectory") == 0) || (clname.find("TTree") == 0) || (clname.find("TNtuple") == 0) ? 1 : 0;
   }

   /** Create element for the browser */
   std::unique_ptr<RBrowserItem> CreateBrowserItem() override
   {
      auto item = std::make_unique<RBrowserTKeyItem>(GetName(), CanHaveChilds());

      item->SetClassName(fKey->GetClassName());

      item->SetIcon(RProvider::GetClassIcon(fKey->GetClassName()));

      return item;
   }

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override;

};

// ===============================================================================================================


/** \class TKeyElement
\ingroup rbrowser

Element representing TKey from TDirectory
*/


class TKeyElement : public RElement {
   TDirectory *fDir{nullptr};
   TKey *fKey{nullptr};

public:
   TKeyElement(TDirectory *dir, TKey *key) : fDir(dir), fKey(key) {}

   virtual ~TKeyElement() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   std::string GetName() const override
   {
      std::string name = fKey->GetName();
      name.append(";");
      name.append(std::to_string(fKey->GetCycle()));
      return name;
   }

   /** Title of RBrowsable (optional) */
   std::string GetTitle() const override { return fKey->GetTitle(); }

   /** Create iterator for childs elements if any
    * Means we should try to browse inside.
    * Either it is directory or some complex object
    * */
   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      std::string clname = fKey->GetClassName();

      if (clname.find("TDirectory") == 0) {
          auto subdir = fDir->GetDirectory(GetName().c_str());
          if (!subdir) return nullptr;
          return std::make_unique<TDirectoryLevelIter>(subdir);
      }

      auto obj = GetObject();

      if (obj) {
         auto elem = Browsable::RProvider::Browse(obj);
         if (elem) return elem->GetChildsIter();
      }

      return nullptr;
   }

   /** Return object associated with TKey, if TDirectory has object of that name it will be returned */
   std::unique_ptr<RHolder> GetObject() override
   {
      std::string clname = fKey->GetClassName();

      auto obj_class = TClass::GetClass(clname.c_str());
      if (!obj_class)
         return nullptr;

      if (obj_class->InheritsFrom(TObject::Class())) {

         TObject *tobj = fDir->FindObject(fKey->GetName());

         if (!tobj) {
            tobj = fKey->ReadObj();

            if (!tobj)
               return nullptr;
         }

         bool owned_by_dir = fDir->FindObject(tobj) == tobj;

         return std::make_unique<RTObjectHolder>(tobj, !owned_by_dir);
      }

      void *obj = fKey->ReadObjectAny(obj_class);
      if (!obj) return nullptr;

      return std::make_unique<RAnyObjectHolder>(obj_class, obj, true);
   }

};

// ==============================================================================================


/////////////////////////////////////////////////////////////////////////////////
/// Return element for current TKey object in TDirectory

std::shared_ptr<RElement> TDirectoryLevelIter::GetElement()
{
   return std::make_shared<TKeyElement>(fDir, fKey);
}

// ==============================================================================================

/** \class TDirectoryElement
\ingroup rbrowser

Element representing TDirectory
*/


class TDirectoryElement : public RElement {
   std::string fFileName;       ///<!   file name
   TDirectory *fDir{nullptr};   ///<!   subdirectory (ifany)

   ///////////////////////////////////////////////////////////////////
   /// Get TDirectory. Checks if parent file is still there. If not, means it was closed outside ROOT

   TDirectory *GetDir()
   {
      if (fDir) {
         if (!gROOT->GetListOfFiles()->FindObject(fDir->GetFile()))
            fDir = nullptr;
      } else if (!fFileName.empty()) {
         fDir = TFile::Open(fFileName.c_str());
      }

      return fDir;
   }

public:

   TDirectoryElement(const std::string &fname, TDirectory *dir = nullptr)
   {
      fFileName = fname;
      fDir = dir;
   }

   virtual ~TDirectoryElement() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   std::string GetName() const override
   {
      if (fDir)
         return fDir->GetName();

      if (!fFileName.empty()) {
         auto pos = fFileName.rfind("/");
         return ((pos == std::string::npos) || (pos > fFileName.length() - 2)) ? fFileName : fFileName.substr(pos + 1);
      }

      return ""s;
   }

   /** Title of RBrowsable (optional) */
   std::string GetTitle() const override
   {
      if (fDir)
         return fDir->GetTitle();

      return "ROOT file "s + fFileName;
   }

   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      auto dir = GetDir();

      return dir ? std::make_unique<TDirectoryLevelIter>(dir) : nullptr;
   }
};


// ==============================================================================================


/** \class RTFileProvider
\ingroup rbrowser

Provides access to ROOT files with extension "root"
Other extensions can be registered
*/


class RTFileProvider : public RProvider {

public:
   RTFileProvider()
   {
      RegisterFile("root", [] (const std::string &fullname) -> std::shared_ptr<RElement> {
         auto f = TFile::Open(fullname.c_str());
         if (!f) return nullptr;

         return std::make_shared<TDirectoryElement>(fullname, f);
      });

      RegisterBrowse(TFile::Class(), [](std::unique_ptr<Browsable::RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TDirectoryElement>("", const_cast<TFile*>(object->Get<TFile>()));
      });

      RegisterBrowse(nullptr, [](std::unique_ptr<Browsable::RHolder> &object) -> std::shared_ptr<RElement> {
         if (object->CanCastTo<TDirectory>())
            return std::make_shared<TDirectoryElement>("", const_cast<TDirectory*>(object->Get<TDirectory>()));
         return nullptr;
      });
   }

} newRTFileProvider ;


