/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <ROOT/RBrowsableTDirectory.hxx>

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


// ===============================================================================================================


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

   void CloseIter()
   {
      fIter.reset(nullptr);
      fKey = nullptr;
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

   std::string GetClassIcon(const std::string &classname)
   {
      if (classname == "TTree" || classname == "TNtuple")
         return "sap-icon://tree"s;
      if (classname == "TDirectory" || classname == "TDirectoryFile")
         return "sap-icon://folder-blank"s;

      return "sap-icon://electronic-medical-record"s;
   }

   /** Create element for the browser */
   std::unique_ptr<RBrowserItem> CreateBrowserItem() override
   {
      auto item = std::make_unique<RBrowserTKeyItem>(GetName(), CanHaveChilds());

      item->SetClassName(fKey->GetClassName());

      item->SetIcon(GetClassIcon(fKey->GetClassName()));

      return item;
   }

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override;

};

// ===============================================================================================================



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

      const TClass *obj_class = TClass::GetClass(clname.c_str());
      if (!obj_class || !obj_class->InheritsFrom(TObject::Class())) return nullptr;

      TObject *obj = fDir->FindObjectAny(fKey->GetName());
      if (!obj) obj = fKey->ReadObj();

      if (obj) {
         printf("Try to browse class %s\n", obj->ClassName());
         // TODO: make clear ownership here, use RObject API here in the future
         auto elem = Browsable::RProvider::Browse(obj->IsA(), obj);
         printf("Got element %p\n", elem.get());
         if (elem) return elem->GetChildsIter();
      }

      return nullptr;
   }

   /** Return TObject depending from kind of requested result */
   std::unique_ptr<RObject> GetObject(bool plain = false) override
   {
      std::string clname = fKey->GetClassName();

      const TClass *obj_class = TClass::GetClass(clname.c_str());
      if (!obj_class) return nullptr;

      if (plain) {

         if (!obj_class->InheritsFrom(TObject::Class())) {
            R__ERROR_HERE("Browserv7") << "Only TObjects can be used for plain reading";
            return nullptr;
         }

         TObject *tobj = fKey->ReadObj();

         if (!tobj)
            return nullptr;

         return std::make_unique<RTObjectHolder>(tobj);
      }

      if (obj_class->InheritsFrom(TObject::Class())) {
         TObject *tobj = fKey->ReadObj();
         if (!tobj)
            return nullptr;
         if (tobj->InheritsFrom(TH1::Class()))
            static_cast<TH1 *>(tobj)->SetDirectory(nullptr);
         if (fDir)
            fDir->Remove(tobj); // remove object while ownership will be delivered to return value
         return std::make_unique<RUnique<TObject>>(tobj);
      }

      void *obj = fKey->ReadObjectAny(obj_class);
      if (!obj) return nullptr;

      return std::make_unique<RAnyObjectHolder>(obj_class, obj, true);
   }

};

// ==============================================================================================


std::shared_ptr<RElement> TDirectoryLevelIter::GetElement()
{
   return std::make_shared<TKeyElement>(fDir, fKey);
}

// ==============================================================================================

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

class RTFileProvider : public RProvider {

public:
   RTFileProvider()
   {
      RegisterFile("root", [] (const std::string &fullname) -> std::shared_ptr<RElement> {
         auto f = TFile::Open(fullname.c_str());
         if (!f) return nullptr;

         return std::make_shared<TDirectoryElement>(fullname, f);
      });
   }

} newRTFileProvider ;


