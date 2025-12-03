/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
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
#include "TColor.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TBufferJSON.h"

#include <sstream>

using namespace std::string_literals;

using namespace ROOT::Browsable;


/** \class TObjectLevelIter
\ingroup rbrowser

Iterator over list of elements, designed for support TBrowser usage
*/

class TObjectLevelIter : public RLevelIter {

   std::vector<std::shared_ptr<RElement>> fElements;

   int fCounter{-1};

public:
   explicit TObjectLevelIter() {}

   ~TObjectLevelIter() override = default;

   void AddElement(std::shared_ptr<RElement> &&elem)
   {
      fElements.emplace_back(std::move(elem));
   }

   auto NumElements() const { return fElements.size(); }

   bool Next() override { return ++fCounter < (int) fElements.size(); }

   // use default implementation for now
   // bool Find(const std::string &name) override { return FindDirEntry(name); }

   std::string GetItemName() const override { return fElements[fCounter]->GetName(); }

   bool CanItemHaveChilds() const override
   {
      std::shared_ptr<TObjectElement> telem = std::dynamic_pointer_cast<TObjectElement>(fElements[fCounter]);
      return telem ? telem->IsFolder() : false;
   }

   /** Create element for the browser */
   std::unique_ptr<RItem> CreateItem() override
   {
      return fElements[fCounter] ? fElements[fCounter]->CreateItem() : nullptr;
   }

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override
   {
      return fElements[fCounter];
   }

   bool Find(const std::string &name, int indx = -1) override
   {
      if ((indx >= 0) && (indx < (int) fElements.size()) && (name == fElements[indx]->GetName())) {
         fCounter = indx;
         return true;
      }

      return RLevelIter::Find(name, -1);
   }

};

// ===============================================================================================================

class TMyBrowserImp : public TBrowserImp {
   TObjectLevelIter *fIter{nullptr};   ///<!  back-reference on iterator
   const TObject *fBrowseObj{nullptr}; ///<!  object which will be browsed
   bool fDuplicated{false};            ///<! is object was duplicated?
   bool fIgnore{false};                 ///<! ignore browsing, used during TBrowser constructor

public:

   TMyBrowserImp(TObjectLevelIter *iter, TObject *obj) : TBrowserImp(nullptr), fIter(iter), fBrowseObj(obj) {}
   ~TMyBrowserImp() override = default;

   void SetIgnore(bool on = true) { fIgnore = on; }

   bool IsDuplicated() const { return fDuplicated; }

   void Add(TObject* obj, const char* name, Int_t) override
   {
      if (fIgnore) return;

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

   void BrowseObj(TObject* obj) override
   {
      if (fIgnore) return;

      Add(obj, nullptr, 0);
   }

};

// ===============================================================================================================

/** \class TCollectionIter
\ingroup rbrowser

Iterator over elements in TCollection
*/

class TCollectionIter : public RLevelIter {

   TIter  fIter;

public:
   explicit TCollectionIter(const TFolder *f) : RLevelIter(), fIter(f->GetListOfFolders()) {}

   explicit TCollectionIter(const TCollection *coll) : RLevelIter(), fIter(coll) {}

   ~TCollectionIter() override = default;

   bool Next() override { return fIter.Next() != nullptr; }

   std::string GetItemName() const override
   {
      auto obj = *fIter;
      if (!obj) return ""s;
      std::string name = obj->GetName();

      if (name.empty()) {
         std::unique_ptr<RHolder> holder = std::make_unique<TObjectHolder>(obj, kFALSE);
         auto elem = RProvider::Browse(holder);
         if (elem) name = elem->CreateItem()->GetName();
      }
      return name;
   }

   /** Check if item can be expanded */
   bool CanItemHaveChilds() const override
   {
      auto obj = *fIter;
      if (!obj || !obj->IsFolder())
         return false;
      return !RProvider::NotShowChilds(obj->IsA());
   }

   /** Create item for current TObject */
   std::unique_ptr<RItem> CreateItem() override
   {
      auto obj = *fIter;
      if (!obj) return RLevelIter::CreateItem();

      std::unique_ptr<RHolder> holder = std::make_unique<TObjectHolder>(obj, kFALSE);

      auto elem = RProvider::Browse(holder);

      if (!elem)
         elem = std::make_shared<TObjectElement>(holder);

      return elem->CreateItem();
   }

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override
   {
      std::unique_ptr<RHolder> holder = std::make_unique<TObjectHolder>(*fIter, kFALSE);

      return RProvider::Browse(holder);
   }

};


// ==============================================================================================

/** \class TFolderElement
\ingroup rbrowser

Browsable element for TFolder
*/


class TFolderElement : public TObjectElement {

public:

   TFolderElement(std::unique_ptr<RHolder> &obj) : TObjectElement(obj) { }

   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      auto folder = fObject->Get<TFolder>();
      if (folder)
         return std::make_unique<TCollectionIter>(folder->GetListOfFolders());

      return TObjectElement::GetChildsIter();
   }

   int GetNumChilds() override
   {
      auto folder = fObject->Get<TFolder>();
      return folder && folder->GetListOfFolders() ? folder->GetListOfFolders()->GetEntries() : 0;
   }
};

// ==============================================================================================

/** \class TCollectionElement
\ingroup rbrowser

Browsable element for TCollection
*/


class TCollectionElement : public TObjectElement {
public:

   TCollectionElement(std::unique_ptr<RHolder> &obj) : TObjectElement(obj) {}

   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      auto coll = fObject->Get<TCollection>();
      if (coll && (coll->GetSize() > 0))
         return std::make_unique<TCollectionIter>(coll);

      return TObjectElement::GetChildsIter();
   }

   int GetNumChilds() override
   {
      auto coll = fObject->Get<TCollection>();
      return coll ? coll->GetSize() : 0;
   }
};

// =================================================================================

////////////////////////////////////////////////////////////////////////////////
/// Constructor with plain TObject* as argument - ownership is not defined

TObjectElement::TObjectElement(TObject *obj, const std::string &name, bool _hide_childs)
{
   SetObject(obj);
   fName = name;
   if (fName.empty())
      fName = fObj->GetName();
   SetHideChilds(_hide_childs);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with std::unique_ptr<RHolder> as argument

TObjectElement::TObjectElement(std::unique_ptr<RHolder> &obj, const std::string &name, bool _hide_childs)
{
   fObject = std::move(obj); // take responsibility
   fObj = const_cast<TObject *>(fObject->Get<TObject>()); // try to cast into TObject

   fName = name;
   if (!fObj)
      fObject.reset();
   else if (fName.empty())
      fName = fObj->GetName();

   SetHideChilds(_hide_childs);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with std::unique_ptr<RHolder> as argument

void TObjectElement::SetObject(TObject *obj)
{
   fObject = std::make_unique<TObjectHolder>(obj);
   fObj = obj;
}

////////////////////////////////////////////////////////////////////////////////
/// Forget object, use when it was deleted behind the scene

void TObjectElement::ForgetObject() const
{
   auto elem = const_cast<TObjectElement *>(this);
   elem->fObj = nullptr;
   if (elem->fObject) {
      elem->fObject->Forget();
      elem->fObject.reset();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Check if object still exists

const TObject *TObjectElement::CheckObject() const
{
   if (!fObj)
      return nullptr;
   if (fObj->IsZombie()) {
      ForgetObject();
      return nullptr;
   }
   return fObj;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if object can have child elements

bool TObjectElement::IsFolder() const
{
   if (IsHideChilds())
      return false;

   return CheckObject() ? fObj->IsFolder() : false;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns name of the TObject

std::string TObjectElement::GetName() const
{
   if (!fName.empty())
      return fName;

   return CheckObject() ? fObj->GetName() : "";
}

////////////////////////////////////////////////////////////////////////////////
/// Returns title of the TObject

std::string TObjectElement::GetTitle() const
{
   return CheckObject() ? fObj->GetTitle() : "";
}

////////////////////////////////////////////////////////////////////////////////
/// Create iterator for childs elements if any

std::unique_ptr<RLevelIter> TObjectElement::GetChildsIter()
{
   if (!IsFolder())
      return nullptr;

   auto iter = std::make_unique<TObjectLevelIter>();

   TMyBrowserImp *imp = new TMyBrowserImp(iter.get(), fObj);

   // ignore browsing during TBrowser constructor, avoid gROOT adding
   imp->SetIgnore(true);

   // must be new, otherwise TBrowser constructor ignores imp
   TBrowser *br = new TBrowser("name", "title", imp);

   imp->SetIgnore(false);

   fObj->Browse(br);

   auto dupl = imp->IsDuplicated();

   delete br; // also will destroy implementation

   if (dupl || (iter->NumElements() == 0)) return nullptr;

   return iter;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns copy of TObject holder - if possible

std::unique_ptr<RHolder> TObjectElement::GetObject()
{
   if (!fObject)
      return nullptr;

   return fObject->Copy();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if holding specified object

bool TObjectElement::IsObject(void *obj)
{
   return (fObj && (fObj == obj));

   // return fObject && (fObject->get_object<TObject>() == obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if object is still valid

bool TObjectElement::CheckValid()
{
   return CheckObject() != nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns class for contained object

const TClass *TObjectElement::GetClass() const
{
   return CheckObject() ? fObj->IsA() : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Provides default action which can be performed with the object

RElement::EActionKind TObjectElement::GetDefaultAction() const
{
   auto cl = GetClass();
   if (!cl) return kActNone;
   std::string clname = cl->GetName();
   if ("TCanvas"s == clname) return kActCanvas;
   if (("TTree"s == clname) || ("TNtuple"s == clname)) return kActTree;
   if (("TGeoManager"s == clname) || (clname.compare(0, 10, "TGeoVolume"s) == 0) || (clname.compare(0, 8, "TGeoNode"s) == 0)) return kActGeom;
   if (RProvider::CanDraw6(cl)) return kActDraw6;
   if (RProvider::CanDraw7(cl)) return kActDraw7;
   if (RProvider::CanHaveChilds(cl)) return kActBrowse;
   return kActNone;
}

////////////////////////////////////////////////////////////////////////////////
/// Create item

std::unique_ptr<RItem> TObjectElement::CreateItem() const
{
   auto obj = CheckObject();
   if (!obj)
      return RElement::CreateItem();

   auto isfolder = !IsHideChilds() && IsFolder();

   auto item = std::make_unique<TObjectItem>(obj->GetName(), isfolder ? -1 : 0);

   if (item->GetName().empty())
      item->SetName(GetName());

   item->SetTitle(obj->GetTitle());
   if (item->GetTitle().empty())
      item->SetTitle(GetTitle());

   item->SetClassName(obj->ClassName());
   item->SetIcon(RProvider::GetClassIcon(obj->IsA(), isfolder));

   auto sz = GetSize();
   if (sz > 0)
      item->SetSize(sz);

   auto tm = GetMTime();
   if (!tm.empty())
      item->SetMTime(tm);

   return item;
}

////////////////////////////////////////////////////////////////////////////////
/// Check object capability

bool TObjectElement::IsCapable(RElement::EActionKind action) const
{
   auto cl = GetClass();
   if (!cl) return false;

   std::string clname = cl->GetName();

   switch(action) {
      case kActBrowse: return RProvider::CanHaveChilds(cl);
      case kActEdit: return true;
      case kActImage:
      case kActDraw6: return RProvider::CanDraw6(cl); // if can draw in TCanvas, can produce image
      case kActDraw7: return RProvider::CanDraw7(cl);
      case kActCanvas: return "TCanvas"s == clname;
      case kActTree: return ("TTree"s == clname) || ("TNtuple"s == clname) ||
                            (clname.compare(0, 7, "TBranch"s) == 0) || (clname.compare(0, 5, "TLeaf"s) == 0);
      case kActGeom: return ("TGeoManager"s == clname) || (clname.compare(0, 10, "TGeoVolume"s) == 0) || (clname.compare(0, 8, "TGeoNode"s) == 0);
      default: return false;
   }

   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates iterator for TCollection object

std::unique_ptr<RLevelIter> TObjectElement::GetCollectionIter(const TCollection *coll)
{
   return std::make_unique<TCollectionIter>(coll);
}

////////////////////////////////////////////////////////////////////////////////
/// Element representing TColor

class TColorElement : public TObjectElement {

public:
   TColorElement(std::unique_ptr<RHolder> &obj) : TObjectElement(obj)
   {
      if (fName.empty()) {
         auto col = fObject->Get<TColor>();
         if (col)
            fName = "Color"s + std::to_string(col->GetNumber());
      }
   }

   EActionKind GetDefaultAction() const override { return kActEdit; }

};

// ==============================================================================================

/** \class RTObjectProvider
\ingroup rbrowser

Provider for all known TObject-based classes
*/

class RTObjectProvider : public RProvider {

public:
   //////////////////////////////////////////////////////////////////////////////////
   // Register TObject-based class with standard browsing/drawing libs

   void RegisterTObject(const std::string &clname, const std::string &iconname, bool can_browse = false, int can_draw = 3, const std::string &drawopt = ""s)
   {
      RegisterClass(clname,
                    iconname,
                    can_browse ? "dflt"s : ""s,
                    can_draw & 1 ? "libROOTObjectDraw6Provider"s : ""s,
                    can_draw & 2 ? "libROOTObjectDraw7Provider"s : ""s,
                    drawopt);
   }

   RTObjectProvider()
   {
      RegisterClass("TTree", "sap-icon://tree", "libROOTBranchBrowseProvider");
      RegisterClass("TNtuple", "sap-icon://tree", "libROOTBranchBrowseProvider");
      RegisterClass("TBranchElement", "sap-icon://e-care", "libROOTBranchBrowseProvider", "libROOTLeafDraw6Provider", "libROOTLeafDraw7Provider");
      RegisterClass("TLeaf", "sap-icon://e-care", ""s, "libROOTLeafDraw6Provider", "libROOTLeafDraw7Provider");
      RegisterClass("TBranch", "sap-icon://e-care", "libROOTBranchBrowseProvider"s, "libROOTLeafDraw6Provider", "libROOTLeafDraw7Provider");
      RegisterClass("TVirtualBranchBrowsable", "sap-icon://e-care", ""s, "libROOTLeafDraw6Provider", "libROOTLeafDraw7Provider");
      RegisterClass("TColor", "sap-icon://palette");
      RegisterClass("TStyle", "sap-icon://badge");

      RegisterTObject("TDirectory", "sap-icon://folder-blank", true, 0);
      RegisterTObject("TClass", "sap-icon://tag-cloud-chart", false, 0);
      RegisterTObject("TQClass", "sap-icon://tag-cloud-chart", false, 0);
      RegisterTObject("TH1", "sap-icon://bar-chart", false, 3, ""s);
      RegisterTObject("TH2", "sap-icon://pixelate", false, 3, "col"s);
      RegisterTObject("TH3", "sap-icon://product");
      RegisterTObject("TProfile", "sap-icon://vertical-bar-chart", false, 3, ""s);
      RegisterTObject("TGraph", "sap-icon://line-chart");
      RegisterTObject("TCanvas", "sap-icon://business-objects-experience", false, 1); // only can use TWebCanvas
      RegisterTObject("TASImage", "sap-icon://picture", false, 1); // only can use TWebCanvas

      RegisterTObject("THStack", "sap-icon://multiple-bar-chart");
      RegisterTObject("TMultiGraph", "sap-icon://multiple-line-chart");

      RegisterTObject("TCollection", "sap-icon://list", true, 0);

      RegisterClass("TGeoManager", "sap-icon://overview-chart", "libROOTGeoBrowseProvider");
      RegisterClass("TGeoVolume", "sap-icon://product", "libROOTGeoBrowseProvider");
      RegisterClass("TGeoNode", "sap-icon://product", "libROOTGeoBrowseProvider");

      RegisterBrowse(TFolder::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TFolderElement>(object);
      });

      RegisterBrowse(TCollection::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TCollectionElement>(object);
      });

      RegisterBrowse(TColor::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TColorElement>(object);
      });

      RegisterBrowse(nullptr, [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         if (object->CanCastTo<TObject>())
            return std::make_shared<TObjectElement>(object, "", NotShowChilds(object->GetClass()));
         return nullptr;
      });

   }

} newRTObjectProvider;
