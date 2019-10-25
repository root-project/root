/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <ROOT/RBrowsableTObject.hxx>

#include "ROOT/RLogger.hxx"

#include "TROOT.h"
#include "TH1.h"
#include "TBrowser.h"
#include "TBrowserImp.h"

using namespace std::string_literals;

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Browsable;


///////////////////////////////////////////////////////////////////////////
/// Return TObject instance with ownership
/// If object is not owned by the holder, it will be cloned (with few exceptions)

void *RTObjectHolder::TakeObject()
{
   auto res = fObj;

   if (fOwner) {
      fObj = nullptr;
      fOwner = false;
   } else if (fObj && !fObj->IsA()->InheritsFrom("TDirectory") && fObj->IsA()->InheritsFrom("TFile")) {
      res = fObj->Clone();
      TH1 *h1 = dynamic_cast<TH1 *>(res);
      if (h1) h1->SetDirectory(nullptr);
   }

   return res;
}


// ===============================================================================================================

/** \class TObjectLevelIter
\ingroup rbrowser

Iterator over keys in TDirectory
*/


class TObjectLevelIter : public RLevelIter {

   std::vector<std::shared_ptr<Browsable::RElement>> fElements;

   int fCounter{-1};

   /** Actually, complete browsing happens here */

   void CloseIter()
   {
   }

   bool NextDirEntry()
   {
      return true;
   }

public:
   explicit TObjectLevelIter() = default;

   virtual ~TObjectLevelIter() = default;

   void AddElement(std::shared_ptr<Browsable::RElement> &&elem)
   {
      fElements.emplace_back(std::move(elem));
   }

   auto NumElements() const { return fElements.size(); }

   bool Reset() override { fCounter = -1; return true; }

   bool Next() override { return ++fCounter < (int) fElements.size(); }

   // use default implementation for now
   // bool Find(const std::string &name) override { return FindDirEntry(name); }

   bool HasItem() const override { return (fCounter >=0) && (fCounter < (int) fElements.size()); }

   std::string GetName() const override { return fElements[fCounter]->GetName(); }

   int CanHaveChilds() const override { return -1; }

   /** Create element for the browser */
   std::unique_ptr<RBrowserItem> CreateBrowserItem() override;

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override
   {
      return fElements[fCounter];
   }

};

// ===============================================================================================================


class TMyBrowserImp : public TBrowserImp {
   TObjectLevelIter &fIter;   ///<!  back-reference on iterat

public:

   TMyBrowserImp(TObjectLevelIter &iter) : TBrowserImp(nullptr), fIter(iter) {}
   virtual ~TMyBrowserImp() = default;

   void Add(TObject* obj, const char* name, Int_t) override;
};

// ===============================================================================================================


class TObjectElement : public RElement {

   std::unique_ptr<Browsable::RHolder> fObject;
   TObject *fObj{nullptr};
   std::string fName;

public:
   TObjectElement(TObject *obj, const std::string &name = "") : fObj(obj), fName(name)
   {
      fObject = std::make_unique<RTObjectHolder>(fObj);
      if (fName.empty())
         fName = fObj->GetName();
   }

   TObjectElement(std::unique_ptr<Browsable::RHolder> &obj, const std::string &name = "")
   {
      fObject = std::move(obj); // take responsibility
      fObj = const_cast<TObject *>(fObject->Get<TObject>()); // try to cast into TObject

      fName = name;
      if (!fObj)
         fObject.reset();
      else if (fName.empty())
         fName = fObj->GetName();
   }


   virtual ~TObjectElement() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   std::string GetName() const override { return fName; }

   /** Title of RBrowsable (optional) */
   std::string GetTitle() const override { return fObj ? fObj->GetTitle() : ""; }

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      if (!fObj) return nullptr;

      auto iter = std::make_unique<TObjectLevelIter>();

      TMyBrowserImp *imp = new TMyBrowserImp(*(iter.get()));

      // must be new, otherwise TBrowser constructor ignores imp
      TBrowser *br = new TBrowser("name", "title", imp);

      fObj->Browse(br);

      delete br;

      // no need to return object itself
      // TODO: make more exact check
      if (iter->NumElements() < 2) return nullptr;

      return iter;
   }

   /** Return copy of TObject holder - if possible */
   std::unique_ptr<RHolder> GetObject() override
   {
      if (!fObject)
         return nullptr;

      return fObject->Copy();
   }

   std::string ClassName() const { return fObj ? fObj->ClassName() : ""; }

};


// ==============================================================================================


void TMyBrowserImp::Add(TObject *obj, const char *name, Int_t)
{
   // printf("Adding object %p %s %s\n", obj, obj->GetName(), obj->ClassName());

   fIter.AddElement(std::make_shared<TObjectElement>(obj, name ? name : ""));
}


// ==============================================================================================


/** Create element for the browser */
std::unique_ptr<RBrowserItem> TObjectLevelIter::CreateBrowserItem()
{
   std::shared_ptr<TObjectElement> elem = std::dynamic_pointer_cast<TObjectElement>(fElements[fCounter]);

   std::string clname = elem->ClassName();
   bool can_have_childs = (clname.find("TDirectory") == 0) || (clname.find("TTree") == 0) || (clname.find("TNtuple") == 0);

   auto item = std::make_unique<RBrowserTObjectItem>(elem->GetName(), can_have_childs ? 1 : 0);

   item->SetClassName(elem->ClassName());

   item->SetIcon(RProvider::GetClassIcon(elem->ClassName()));

   return item;
}


// ==============================================================================================


class RTObjectProvider : public RProvider {

public:
   RTObjectProvider()
   {
      RegisterBrowse(nullptr, [](std::unique_ptr<Browsable::RHolder> &object) -> std::shared_ptr<RElement> {
         if (object->CanCastTo<TObject>())
            return std::make_shared<TObjectElement>(object);

         return nullptr;

      });
   }

} newRTObjectProvider;
