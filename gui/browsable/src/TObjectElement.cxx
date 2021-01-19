/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/TObjectElement.hxx>
#include <ROOT/Browsable/RProvider.hxx>
#include <ROOT/Browsable/TObjectHolder.hxx>
#include <ROOT/Browsable/TObjectItem.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>

#include <ROOT/RLogger.hxx>

#include "TBrowser.h"
#include "TBrowserImp.h"
#include "TFolder.h"
#include "TList.h"
#include "TDirectory.h"
#include "TBufferJSON.h"

#include <sstream>

using namespace std::string_literals;

using namespace ROOT::Experimental::Browsable;

/** \class TObjectLevelIter
\ingroup rbrowser

Iterator over list of elements, designed for support TBrowser usage
*/

class TObjectLevelIter : public RLevelIter {

   std::vector<std::shared_ptr<RElement>> fElements;

   int fCounter{-1};

public:
   explicit TObjectLevelIter() {}

   virtual ~TObjectLevelIter() = default;

   void AddElement(std::shared_ptr<RElement> &&elem)
   {
      fElements.emplace_back(std::move(elem));
   }

   auto NumElements() const { return fElements.size(); }

   bool Next() override { return ++fCounter < (int) fElements.size(); }

   // use default implementation for now
   // bool Find(const std::string &name) override { return FindDirEntry(name); }

   std::string GetItemName() const override { return fElements[fCounter]->GetName(); }

   int GetNumItemChilds() const override
   {
      std::shared_ptr<TObjectElement> telem = std::dynamic_pointer_cast<TObjectElement>(fElements[fCounter]);
      if (!telem) return 0;
      return telem->IsFolder() ? -1 : 0;
   }

   /** Create element for the browser */
   std::unique_ptr<RItem> CreateItem() override;

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override
   {
      return fElements[fCounter];
   }

};

// ===============================================================================================================

class TMyBrowserImp : public TBrowserImp {
   TObjectLevelIter *fIter{nullptr};   ///<!  back-reference on iterator
   const TObject *fBrowseObj{nullptr}; ///<!  object which wil be browsed
   bool fDuplicated{false};            ///<! is object was duplicated?

public:

   TMyBrowserImp(TObjectLevelIter *iter, TObject *obj) : TBrowserImp(nullptr), fIter(iter), fBrowseObj(obj) {}
   virtual ~TMyBrowserImp() = default;

   void Add(TObject* obj, const char* name, Int_t) override;

   bool IsDuplicated() const { return fDuplicated; }
};

// ===============================================================================================================


TObjectElement::TObjectElement(TObject *obj, const std::string &name) : fObj(obj), fName(name)
{
   fObject = std::make_unique<TObjectHolder>(fObj);
   if (fName.empty())
      fName = fObj->GetName();
}

TObjectElement::TObjectElement(std::unique_ptr<RHolder> &obj, const std::string &name)
{
   fObject = std::move(obj); // take responsibility
   fObj = const_cast<TObject *>(fObject->Get<TObject>()); // try to cast into TObject

   fName = name;
   if (!fObj)
      fObject.reset();
   else if (fName.empty())
      fName = fObj->GetName();
}

TObjectElement::~TObjectElement()
{
}

std::string TObjectElement::GetName() const
{
   if (!fName.empty()) return fName;
   return fObj ? fObj->GetName() : "";
}

/** Title of TObject */
std::string TObjectElement::GetTitle() const
{
   return fObj ? fObj->GetTitle() : "";
}

/** Returns IsFolder of contained object */
bool TObjectElement::IsFolder()
{
   return fObj ? fObj->IsFolder() : false;
}

/** Create iterator for childs elements if any */
std::unique_ptr<RLevelIter> TObjectElement::GetChildsIter()
{
   if (!IsFolder()) return nullptr;

   auto iter = std::make_unique<TObjectLevelIter>();

   TMyBrowserImp *imp = new TMyBrowserImp(iter.get(), fObj);

   // must be new, otherwise TBrowser constructor ignores imp
   TBrowser *br = new TBrowser("name", "title", imp);

   fObj->Browse(br);

   auto dupl = imp->IsDuplicated();

   delete br; // also will destroy implementaion

   if (dupl || (iter->NumElements() == 0)) return nullptr;

   return iter;
}

/** Return copy of TObject holder - if possible */
std::unique_ptr<RHolder> TObjectElement::GetObject()
{
   if (!fObject)
      return nullptr;

   return fObject->Copy();
}

/** Return class for contained object */
const TClass *TObjectElement::GetClass() const
{
   return fObj ? fObj->IsA() : nullptr;
}


// ==============================================================================================


void TMyBrowserImp::Add(TObject *obj, const char *name, Int_t)
{
   // prevent duplication of object itself - ignore such browsing
   if (fBrowseObj == obj) fDuplicated = true;
   if (fDuplicated) return;

   std::unique_ptr<RHolder> holder = std::make_unique<TObjectHolder>(obj);

   std::shared_ptr<RElement> elem = RProvider::Browse(holder);

   if (name && *name) {
      std::shared_ptr<TObjectElement> telem = std::dynamic_pointer_cast<TObjectElement>(elem);
      if (telem) telem->SetName(name);
   }

   fIter->AddElement(std::move(elem));
}


// ==============================================================================================


///////////////////////////////////////////////////////////////
/// Create element for the browser

std::unique_ptr<RItem> TObjectLevelIter::CreateItem()
{
   std::shared_ptr<TObjectElement> elem = std::dynamic_pointer_cast<TObjectElement>(fElements[fCounter]);
   // should never happen
   if (!elem) return nullptr;

   auto cl = elem->GetClass();

   auto nchilds = elem->GetNumChilds();

   auto item = std::make_unique<TObjectItem>(elem->GetName(), nchilds);

   item->SetClassName(cl ? cl->GetName() : "");

   item->SetIcon(RProvider::GetClassIcon(cl));

   return item;
}


// ==============================================================================================

class TFolderElement : public TObjectElement {

public:

   TFolderElement(std::unique_ptr<RHolder> &obj) : TObjectElement(obj) {}

   std::unique_ptr<RLevelIter> GetChildsIter() override;

   int GetNumChilds() override;
};

class TCollectionElement : public TObjectElement {
public:

   TCollectionElement(std::unique_ptr<RHolder> &obj) : TObjectElement(obj) {}

   std::unique_ptr<RLevelIter> GetChildsIter() override;

   int GetNumChilds() override;
};



class TCollectionIter : public RLevelIter {

   TIter  fIter;

public:
   explicit TCollectionIter(const TFolder *f) : RLevelIter(), fIter(f->GetListOfFolders()) {};

   explicit TCollectionIter(const TCollection *coll) : RLevelIter(), fIter(coll) {};

   virtual ~TCollectionIter() = default;

   bool Next() override { return fIter.Next() != nullptr; }

   // use default implementation for now
   // bool Find(const std::string &name) override { return FindDirEntry(name); }

   std::string GetItemName() const override { return (*fIter)->GetName(); }

   int GetNumItemChilds() const override
   {
      TObject *obj = *fIter;
      if (!obj) return 0;
      return obj->IsFolder() ? -1 : 0;
   }

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override
   {
      std::unique_ptr<RHolder> holder = std::make_unique<TObjectHolder>(*fIter, kFALSE);

      return RProvider::Browse(holder);
   }
};

//////////////////////////////////////////////////////////////////////////////////////
/// Provides iterator for TFolder

std::unique_ptr<RLevelIter> TFolderElement::GetChildsIter()
{
  auto folder = fObject->Get<TFolder>();
  if (folder)
     return std::make_unique<TCollectionIter>(folder->GetListOfFolders());

  return TObjectElement::GetChildsIter();
}

//////////////////////////////////////////////////////////////////////////////////////
/// Returns number of childs in the TFolder

int TFolderElement::GetNumChilds()
{
   auto folder = fObject->Get<TFolder>();
   return folder && folder->GetListOfFolders() ? folder->GetListOfFolders()->GetEntries() : 0;
}


//////////////////////////////////////////////////////////////////////////////////////
/// Provides iterator for generic TCollecion

std::unique_ptr<RLevelIter> TCollectionElement::GetChildsIter()
{
   auto coll = fObject->Get<TCollection>();
   if (coll && (coll->GetSize() > 0))
      return std::make_unique<TCollectionIter>(coll);

   return TObjectElement::GetChildsIter();
}

//////////////////////////////////////////////////////////////////////////////////////
/// Returns number of childs in the TFolder

int TCollectionElement::GetNumChilds()
{
   auto coll = fObject->Get<TCollection>();
   return coll ? coll->GetSize() : 0;
}

// ==============================================================================================

class RTObjectProvider : public RProvider {

public:
   //////////////////////////////////////////////////////////////////////////////////
   // Register TObject-based class with standard browsing/drawing libs

   void RegisterTObject(const std::string &clname, const std::string &iconname, bool can_browse = false, bool can_draw = true)
   {
      RegisterClass(clname, iconname, can_browse ? "dflt"s : ""s,
                                      can_draw ? "libROOTObjectDraw6Provider"s : ""s,
                                      can_draw ? "libROOTObjectDraw7Provider"s : ""s);
   }

   RTObjectProvider()
   {
      RegisterTObject("TTree", "sap-icon://tree", true, false);
      RegisterTObject("TNtuple", "sap-icon://tree", true, false);
      RegisterClass("TBranchElement", "sap-icon://e-care", "libROOTBranchBrowseProvider", "libROOTLeafDraw6Provider", "libROOTLeafDraw7Provider");
      RegisterClass("TLeaf", "sap-icon://e-care", ""s, "libROOTLeafDraw6Provider", "libROOTLeafDraw7Provider");

      RegisterTObject("TDirectory", "sap-icon://folder-blank", true, false);
      RegisterTObject("TH1", "sap-icon://vertical-bar-chart");
      RegisterTObject("TH2", "sap-icon://pixelate");
      RegisterTObject("TProfile", "sap-icon://vertical-bar-chart");
      RegisterTObject("TGraph", "sap-icon://line-chart");

      RegisterTObject("TGeoManager", "sap-icon://tree", true, false);

      RegisterBrowse(TFolder::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TFolderElement>(object);
      });

      RegisterBrowse(TCollection::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TCollectionElement>(object);
      });

      RegisterBrowse(nullptr, [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         if (object->CanCastTo<TObject>())
            return std::make_shared<TObjectElement>(object);
         return nullptr;
      });
   }

} newRTObjectProvider;
