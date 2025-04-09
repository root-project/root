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
#include <ROOT/Browsable/TObjectElement.hxx>
#include <ROOT/Browsable/TObjectItem.hxx>
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

using namespace ROOT::Browsable;


/** \class TDirectoryLevelIter
\ingroup rbrowser

Iterator over keys in TDirectory
*/


class TDirectoryLevelIter : public RLevelIter {
   TDirectory *fDir{nullptr};         ///<! current directory handle
   std::unique_ptr<TIterator> fIter;  ///<! created iterator
   bool fKeysIter{true};              ///<! iterating over keys list (default)
   bool fOnlyLastCycle{false};        ///<! show only last cycle in list of keys
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
         // exclude object with duplicated name as keys
         while (fObj) {
            if (!fDir->GetListOfKeys()->FindObject(fObj->GetName()))
               break;
            fObj = fIter->Next();
         }
         if (!fObj) {
            fIter.reset();
            return false;
         }

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
               fOnlyLastCycle = true;
            else if (svalue == "no")
               fOnlyLastCycle = false;
            else
               R__LOG_ERROR(ROOT::BrowsableLog()) << "WebGui.LastCycle must be yes or no";
         }
      }

      CreateIter();
   }

   ~TDirectoryLevelIter() override = default;

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

   /** Create item for the client */
   std::unique_ptr<RItem> CreateItem() override
   {
      if (!fKeysIter && fObj) {
         std::unique_ptr<RHolder> holder = std::make_unique<TObjectHolder>(fObj, kFALSE);

         auto elem = RProvider::Browse(holder);

         return elem ? elem->CreateItem() : nullptr;
      }

      auto item = GetDirElement(false)->CreateItem();
      item->SetName(fCurrentName);
      return item;
   }

   std::shared_ptr<RElement> GetDirElement(bool read_dir);

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override { return GetDirElement(true); }
};

// ===============================================================================================================

/** \class TDirectoryElement
\ingroup rbrowser

Element representing TDirectory
*/

class TDirectoryElement : public TObjectElement {
   std::string fFileName;       ///<!   file name
   bool fIsFile{false};         ///<!   is TFile instance registered in global list of files

protected:

   const TObject *CheckObject() const override
   {
      // during TROOT destructor just forget about file reference
      if (!gROOT || gROOT->TestBit(TObject::kInvalidObject)) {
         ForgetObject();
         return nullptr;
      }

      if (!TObjectElement::CheckObject())
         return nullptr;

      if (fIsFile) {
         if (!gROOT->GetListOfFiles()->FindObject(fObj))
            ForgetObject();
      } else if (!gROOT->GetListOfFiles()->FindObject(((TDirectory *) fObj)->GetFile()))
         ForgetObject();

      return fObj;
   }

   TDirectory *GetDir() const
   {
      if (!CheckObject() && fIsFile && fFileName.empty())
         (const_cast<TDirectoryElement *>(this))->SetObject(TFile::Open(fFileName.c_str()));

      return dynamic_cast<TDirectory *>(fObj);
   }

   TFile *GetFile() const
   {
      if (!fIsFile)
         return nullptr;

      return dynamic_cast<TFile *>(GetDir());
   }


public:

   TDirectoryElement(const std::string &fname, TDirectory *dir = nullptr, bool isfile = false) : TObjectElement(dir)
   {
      fFileName = fname;
      fIsFile = isfile;
      if (fIsFile && fObj && !gROOT->GetListOfFiles()->FindObject(fObj)) {
         fIsFile = false;
         ForgetObject();
      }
   }

   ~TDirectoryElement() override = default;

   /** Name of TDirectoryElement */
   std::string GetName() const override
   {
      if (CheckObject())
         return fObj->GetName();

      if (!fFileName.empty()) {
         auto pos = fFileName.rfind("/");
         return ((pos == std::string::npos) || (pos > fFileName.length() - 2)) ? fFileName : fFileName.substr(pos + 1);
      }

      return ""s;
   }

   /** Title of TDirectoryElement */
   std::string GetTitle() const override
   {
      if (CheckObject())
         return fObj->GetTitle();

      return "ROOT file "s + fFileName;
   }

   bool IsFolder() const override { return true; }

   /** Provide iterator over TDirectory */
   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      auto dir = GetDir();

      return dir ? std::make_unique<TDirectoryLevelIter>(dir) : nullptr;
   }

   /** Get default action - browsing for the TFile/TDirectory */
   EActionKind GetDefaultAction() const override { return kActBrowse; }

   /** Select directory as active */
   bool cd() override
   {
      auto dir = GetDir();
      if (dir) {
         dir->cd();
         return true;
      }
      return false;
   }

   /** Size of TDirectory */
   Long64_t GetSize() const override
   {
      auto f = GetFile();
      if (f) return f->GetSize();
      return -1;
   }

   std::string GetMTime() const override
   {
      auto f = GetFile();
      if (f) return f->GetModificationDate().AsSQLString();
      return ""s;
   }

   std::string GetContent(const std::string &kind) override
   {
      if (GetContentKind(kind) == kFileName)
         return fFileName;

      return ""s;
   }

};

// ===============================================================================================================


/** \class TKeyElement
\ingroup rbrowser

Element representing TKey from TDirectory
*/

class TKeyElement : public TDirectoryElement {
   std::string fKeyName, fKeyTitle, fKeyClass, fKeyMTime;
   Short_t fKeyCycle{0};
   Long64_t fKeyObjSize{-1};
   std::shared_ptr<RElement> fElement; ///<! holder of read object

   std::string GetMTime() const override { return fKeyMTime; }

   Long64_t GetSize() const override  { return fKeyObjSize; }

public:
   TKeyElement(TDirectory *dir, TKey *key) : TDirectoryElement("", dir, false)
   {
      fKeyName = key->GetName();
      fKeyTitle = key->GetTitle();
      fKeyCycle = key->GetCycle();
      fKeyClass = key->GetClassName();
      fKeyMTime = key->GetDatime().AsSQLString();
      fKeyObjSize = key->GetNbytes();
   }

   ~TKeyElement() override = default;

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
         auto dir = GetDir();
         if (!dir) return nullptr;

         auto subdir = dir->GetDirectory(fKeyName.c_str());
         if (!subdir)
            subdir = dir->GetDirectory(GetName().c_str());
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
         R__LOG_ERROR(ROOT::BrowsableLog()) << "Class " << fKeyClass << " does not have dictionary, object " << fKeyName << " cannot be read";
         return nullptr;
      }
      auto dir = GetDir();
      if (!dir)
         return nullptr;

      std::string namecycle = fKeyName + ";"s + std::to_string(fKeyCycle);

      void *obj = dir->GetObjectChecked(namecycle.c_str(), obj_class);

      if (!obj)
         return nullptr;

      TObject *tobj = (TObject *) obj_class->DynamicCast(TObject::Class(), obj);

      if (tobj) {
         bool in_dir = dir->FindObject(tobj) != nullptr,
              special_class = (fKeyClass == "TGeoManager"s) || (fKeyClass == "TTree"s) || (fKeyClass == "TNtuple"s);

         if (in_dir && !special_class)
            dir->Remove(tobj);

         return std::make_unique<TObjectHolder>(tobj, !special_class);
      }

      return std::make_unique<RAnyObjectHolder>(obj_class, obj, true);
   }

   EActionKind GetDefaultAction() const override
   {
      if (fElement)
         return fElement->GetDefaultAction();

      if (fKeyClass.empty()) return kActNone;
      if ((fKeyClass == "TCanvas"s) || (fKeyClass == "ROOT::Experimental::RCanvas"s)) return kActCanvas;
      if ((fKeyClass == "TTree"s) || (fKeyClass == "TNtuple"s)) return kActTree;
      if (fKeyClass == "TGeoManager"s) return kActGeom;
      if (RProvider::CanDraw6(fKeyClass)) return kActDraw6;
      if (RProvider::CanDraw7(fKeyClass)) return kActDraw7;
      if (RProvider::CanHaveChilds(fKeyClass)) return kActBrowse;
      return kActNone;
   }

   bool IsFolder() const override
   {
      if (fElement)
         return fElement->IsFolder();

      if (!fKeyClass.empty()) {
         if (RProvider::CanHaveChilds(fKeyClass))
            return true;
         auto cl = TClass::GetClass(fKeyClass.c_str(), kFALSE, kTRUE);
         return RProvider::CanHaveChilds(cl);
      }

      return false;
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
         case kActTree: return (fKeyClass == "TTree"s) || (fKeyClass == "TNtuple"s);
         case kActGeom: return (fKeyClass == "TGeoManager"s);
         default: return false;
      }

      return false;
   }

   std::unique_ptr<RItem> CreateItem() const override
   {
      if (fElement)
         return fElement->CreateItem();

      bool is_folder = IsFolder();

      auto item = std::make_unique<TObjectItem>(GetName(), is_folder  ? -1 : 0);
      item->SetTitle(fKeyTitle);
      item->SetClassName(fKeyClass);
      item->SetIcon(RProvider::GetClassIcon(fKeyClass, is_folder));
      item->SetSize(fKeyObjSize);
      item->SetMTime(fKeyMTime);

      return item;
   }

};

// ==============================================================================================

/////////////////////////////////////////////////////////////////////////////////
/// Return element for current TKey object in TDirectory

std::shared_ptr<RElement> TDirectoryLevelIter::GetDirElement(bool read_dir)
{
   if (!fKeysIter && fObj)
      return std::make_shared<TObjectElement>(fObj);

   if ("ROOT::RNTuple"s == fKey->GetClassName())
      return RProvider::BrowseNTuple(fKey->GetName(), fDir->GetFile()->GetName());

   std::string key_class = fKey->GetClassName();
   if (read_dir && (key_class.find("TDirectory") == 0)) {
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
         return std::make_shared<TDirectoryElement>(fullname, f, true);
      });

      RegisterBrowse(TFile::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TDirectoryElement>("", const_cast<TFile*>(object->Get<TFile>()), true);
      });

      RegisterBrowse(TDirectory::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TDirectoryElement>("", const_cast<TDirectory*>(object->Get<TDirectory>()));
      });
   }

} newRTFileProvider;
