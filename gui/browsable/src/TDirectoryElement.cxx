/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <ROOT/Browsable/RAnyObjectHolder.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>
#include <ROOT/Browsable/RProvider.hxx>
#include <ROOT/Browsable/TObjectHolder.hxx>
#include <ROOT/Browsable/TKeyItem.hxx>
#include <ROOT/Browsable/TObjectItem.hxx>
#include <ROOT/Browsable/TObjectElement.hxx>
#include <ROOT/Browsable/RHolder.hxx>

#include <ROOT/RLogger.hxx>

#include "TKey.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "TFile.h"
#include "TClass.h"
#include "TEnv.h"

#include <cstring>
#include <string>

using namespace std::string_literals;

using namespace ROOT::Experimental::Browsable;


/** \class TDirectoryLevelIter
\ingroup rbrowser

Iterator over keys in TDirectory
*/


class TDirectoryLevelIter : public RLevelIter {
   TDirectory *fDir{nullptr};         ///<! current directory handle
   std::unique_ptr<TIterator> fIter;  ///<! created iterator
   Bool_t fKeysIter{kTRUE};           ///<! iterating over keys list (default)
   Bool_t fOnlyLastCycle{kFALSE};     ///<! show only last cycle in list of keys
   TKey *fKey{nullptr};               ///<! currently selected key
   TObject *fObj{nullptr};            ///<! currently selected object
   std::string fCurrentName;          ///<! current key name

   bool CreateIter()
   {
      if (!fDir) return false;
      fObj = nullptr;
      fKey = nullptr;
      auto lst = fDir->GetListOfKeys();
      if (lst->GetSize() == 0) {
         auto olst = fDir->GetList();
         if (olst->GetSize() > 0) {
            fKeysIter = false;
            fIter.reset(olst->MakeIterator());
            return true;
         }
      }
      fKeysIter = true;
      fIter.reset(lst->MakeIterator());
      return true;
   }

   bool NextDirEntry()
   {
      fCurrentName.clear();
      if (!fIter) return false;

      fObj = fIter->Next();
      if (!fObj) {
         fIter.reset();
         if (!fKeysIter || !fDir)
            return false;
         fKeysIter = false;
         fIter.reset(fDir->GetList()->MakeIterator());
         fObj = fIter->Next();
         if (!fObj) {
            fIter.reset();
            return false;
         }
      }
      if (!fKeysIter) {
         fCurrentName = fObj->GetName();
         return true;
      }

      while(true) {

         fKey = dynamic_cast<TKey *>(fObj);

         if (!fKey) {
            fIter.reset();
            return false;
         }

         if (!fOnlyLastCycle) break;

         TIter iter(fDir->GetListOfKeys());
         TKey *key = nullptr;
         bool found_newer = false;
         while ((key = dynamic_cast<TKey*>(iter())) != nullptr) {
            if ((key != fKey) && !strcmp(key->GetName(), fKey->GetName()) && (key->GetCycle() > fKey->GetCycle())) {
               found_newer = true;
               break;
            }
         }

         if (!found_newer) break;

         fObj = fIter->Next();
      }

      fCurrentName = fKey->GetName();
      fCurrentName.append(";");
      fCurrentName.append(std::to_string(fKey->GetCycle()));

      return true;
   }

public:
   explicit TDirectoryLevelIter(TDirectory *dir) : fDir(dir)
   {
      const char *undef = "<undefined>";
      const char *value = gEnv->GetValue("WebGui.LastCycle", undef);
      if (value) {
         std::string svalue = value;
         if (svalue != undef) {
            if (svalue == "yes")
               fOnlyLastCycle = kTRUE;
            else if (svalue == "no")
               fOnlyLastCycle = kFALSE;
            else
               R__LOG_ERROR(ROOT::Experimental::BrowsableLog()) << "WebGui.LastCycle must be yes or no";
         }
      }

      CreateIter();
   }

   virtual ~TDirectoryLevelIter() = default;

   bool Next() override { return NextDirEntry(); }

   // use default implementation for now
   // bool Find(const std::string &name) override { return FindDirEntry(name); }

   std::string GetItemName() const override { return fCurrentName; }

   bool CanItemHaveChilds() const override
   {
      if (!fKeysIter && fObj)
         return RProvider::CanHaveChilds(fObj->IsA());

      if (fKeysIter && fKey) {
         if (RProvider::CanHaveChilds(fKey->GetClassName()))
            return true;
         auto cl = TClass::GetClass(fKey->GetClassName(), kFALSE, kTRUE);
         return RProvider::CanHaveChilds(cl);
      }
      return false;
   }

   /** Create element for the browser */
   std::unique_ptr<RItem> CreateItem() override
   {
      if (!fKeysIter && fObj) {
         auto item = std::make_unique<TObjectItem>(GetItemName(), CanItemHaveChilds() ? -1 : 0);
         item->SetClassName(fObj->IsA()->GetName());
         item->SetIcon(RProvider::GetClassIcon(fObj->IsA()->GetName()));
         item->SetTitle(fObj->GetTitle());
         return item;
      }

      auto item = std::make_unique<TKeyItem>(GetItemName(), CanItemHaveChilds() ? -1 : 0);
      item->SetClassName(fKey->GetClassName());
      item->SetIcon(RProvider::GetClassIcon(fKey->GetClassName()));
      item->SetTitle(fKey->GetTitle());
      item->SetSize(std::to_string(fKey->GetNbytes()));
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
   std::string fKeyName;
   std::string fKeyTitle;
   Short_t fKeyCycle{0};
   std::string fKeyClass;
   std::shared_ptr<RElement> fElement; ///<! holder of read object

public:
   TKeyElement(TDirectory *dir, TKey *key) : fDir(dir)
   {
      fKeyName = key->GetName();
      fKeyTitle = key->GetTitle();
      fKeyCycle = key->GetCycle();
      fKeyClass = key->GetClassName();
   }

   virtual ~TKeyElement() = default;

   /** Name of TKeyElement, includes key cycle */
   std::string GetName() const override
   {
      if (fElement)
         return fElement->GetName();
      std::string name = fKeyName;
      name.append(";");
      name.append(std::to_string(fKeyCycle));

      return name;
   }

   /** Title of TKeyElement (optional) */
   std::string GetTitle() const override
   {
      if (fElement)
         return fElement->GetTitle();

      return fKeyTitle;
   }

   /** Create iterator for childs elements if any
     * Means we should try to browse inside.
     * Either it is directory or some complex object */
   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      if (fElement)
         return fElement->GetChildsIter();

      if (fKeyClass.find("TDirectory") == 0) {
         auto subdir = fDir->GetDirectory(fKeyName.c_str());
         if (!subdir)
            subdir = fDir->GetDirectory(GetName().c_str());
         if (!subdir) return nullptr;
         return std::make_unique<TDirectoryLevelIter>(subdir);
      }

      auto obj = GetObject();

      if (obj)
         fElement = RProvider::Browse(obj);

      if (fElement)
         return fElement->GetChildsIter();

      return nullptr;
   }

   /** Return object associated with the TKey */
   std::unique_ptr<RHolder> GetObject() override
   {
      if (fElement)
         return fElement->GetObject();

      auto obj_class = TClass::GetClass(fKeyClass.c_str());
      if (!obj_class)
         return nullptr;

      if (!obj_class->HasDictionary()) {
         R__LOG_ERROR(ROOT::Experimental::BrowsableLog()) << "Class " << fKeyClass << " does not have dictionary, object " << fKeyName << " cannot be read";
         return nullptr;
      }

      std::string namecycle = fKeyName + ";"s + std::to_string(fKeyCycle);

      void *obj = fDir->GetObjectChecked(namecycle.c_str(), obj_class);

      if (!obj)
         return nullptr;

      TObject *tobj = (TObject *) obj_class->DynamicCast(TObject::Class(), obj);

      if (tobj) {
         bool owned_by_dir = (fDir->FindObject(tobj) == tobj) || (fKeyClass == "TGeoManager");

         return std::make_unique<TObjectHolder>(tobj, !owned_by_dir);
      }

      return std::make_unique<RAnyObjectHolder>(obj_class, obj, true);
   }


   EActionKind GetDefaultAction() const override
   {
      if (fElement)
         return fElement->GetDefaultAction();

      if (fKeyClass.empty()) return kActNone;
      if ((fKeyClass == "TCanvas"s) || (fKeyClass == "ROOT::Experimental::RCanvas"s)) return kActCanvas;
      if (fKeyClass == "TGeoManager"s) return kActGeom;
      if (RProvider::CanDraw6(fKeyClass)) return kActDraw6;
      if (RProvider::CanDraw7(fKeyClass)) return kActDraw7;
      if (RProvider::CanHaveChilds(fKeyClass)) return kActBrowse;
      return kActNone;
   }

   bool IsCapable(EActionKind action) const override
   {
      if (fElement)
         return fElement->IsCapable(action);

      if (fKeyClass.empty()) return false;

      switch(action) {
         case kActBrowse: {
            if (RProvider::CanHaveChilds(fKeyClass))
               return true;
            return RProvider::CanHaveChilds(TClass::GetClass(fKeyClass.c_str(), kFALSE, kTRUE));
         }
         case kActEdit: return true;
         case kActImage:
         case kActDraw6: {
            // if can draw in TCanvas, can produce image
            if (RProvider::CanDraw6(fKeyClass))
               return true;
            return RProvider::CanDraw6(TClass::GetClass(fKeyClass.c_str(), kFALSE, kTRUE));
         }
         case kActDraw7: {
            if (RProvider::CanDraw7(fKeyClass))
               return true;
            return RProvider::CanDraw7(TClass::GetClass(fKeyClass.c_str(), kFALSE, kTRUE));
         }
         case kActCanvas: return (fKeyClass == "TCanvas"s) || (fKeyClass == "ROOT::Experimental::RCanvas"s);
         case kActGeom: return (fKeyClass == "TGeoManager"s);
         default: return false;
      }

      return false;
   }

};

// ==============================================================================================

/** \class TDirectoryElement
\ingroup rbrowser

Element representing TDirectory
*/


class TDirectoryElement : public RElement {
   std::string fFileName;       ///<!   file name
   TDirectory *fDir{nullptr};   ///<!   sub-directory (if any)

   ///////////////////////////////////////////////////////////////////
   /// Get TDirectory. Checks if parent file is still there. If not, means it was closed outside ROOT

   TDirectory *GetDir()
   {
      if (fDir) {
         if (fDir->IsZombie())
            fDir = nullptr;
         else if (!gROOT->GetListOfFiles()->FindObject(fDir->GetFile()))
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

   /** Name of TDirectoryElement */
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

   /** Title of TDirectoryElement */
   std::string GetTitle() const override
   {
      if (fDir)
         return fDir->GetTitle();

      return "ROOT file "s + fFileName;
   }

   /** Provide iterator over TDirectory */
   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      auto dir = GetDir();

      return dir ? std::make_unique<TDirectoryLevelIter>(dir) : nullptr;
   }

   /** Get default action - browsing for the TFile/TDirectory*/
   EActionKind GetDefaultAction() const override { return kActBrowse; }

   /** Select directory as active */
   bool cd() override
   {
      if (fDir && !fDir->IsZombie()) {
         fDir->cd();
         return true;
      }
      return false;
   }

};


class TObjectIndDirElement : public TObjectElement {
protected:
   TDirectory   *fDir{nullptr};

   bool CheckObject() const override
   {
      if (!fDir || !fObj)
        return false;

      if (!fDir->IsZombie() || !fDir->FindObject(fObj)) {
         auto self = const_cast<TObjectIndDirElement *>(this);
         self->fDir = nullptr;
         self->fObj = nullptr;
         self->fObject.reset();
         return false;
      }
      return true;
   }

public:
   TObjectIndDirElement(TDirectory *dir, TObject *obj) : TObjectElement(obj), fDir(dir) {}
};

// ==============================================================================================

/////////////////////////////////////////////////////////////////////////////////
/// Return element for current TKey object in TDirectory

std::shared_ptr<RElement> TDirectoryLevelIter::GetElement()
{
   if (!fKeysIter && fObj)
      return std::make_shared<TObjectElement>(fObj);

   if ("ROOT::Experimental::RNTuple"s == fKey->GetClassName())
      return RProvider::BrowseNTuple(fKey->GetName(), fDir->GetFile()->GetName());

   std::string key_class = fKey->GetClassName();
   if (key_class.find("TDirectory") == 0) {
      auto subdir = fDir->GetDirectory(fKey->GetName());
      if (subdir) return std::make_shared<TDirectoryElement>("", subdir);
   }

   return std::make_shared<TKeyElement>(fDir, fKey);
}



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
         auto f = dynamic_cast<TFile *> (gROOT->GetListOfFiles()->FindObject(fullname.c_str()));
         if (!f) f = TFile::Open(fullname.c_str());
         if (!f) return nullptr;
         return std::make_shared<TDirectoryElement>(fullname, f);
      });

      RegisterBrowse(TFile::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TDirectoryElement>("", const_cast<TFile*>(object->Get<TFile>()));
      });

      RegisterBrowse(TDirectory::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TDirectoryElement>("", const_cast<TDirectory*>(object->Get<TDirectory>()));
      });
   }

} newRTFileProvider;


