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


// ===============================================================================================================



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
   TObjectLevelIter *fIter;

public:

   TMyBrowserImp(TObjectLevelIter *iter) : TBrowserImp(nullptr), fIter(iter) {}
   virtual ~TMyBrowserImp() = default;

   void Add(TObject* obj, const char* name, Int_t) override;
};

// ===============================================================================================================


class TObjectElement : public RElement {
   TObject *fObj{nullptr};
   std::string fName;

public:
   TObjectElement(TObject *obj, const std::string &name = "") : fObj(obj), fName(name)
   {
      if (fName.empty())
         fName = fObj->GetName();
   }

   virtual ~TObjectElement() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   std::string GetName() const override { return fName; }

   /** Title of RBrowsable (optional) */
   std::string GetTitle() const override { return fObj->GetTitle(); }

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      auto iter = std::make_unique<TObjectLevelIter>();

      TMyBrowserImp *imp = new TMyBrowserImp(iter.get());

      // must be new, otherwise TBrowser constructor ignores imp
      TBrowser *br = new TBrowser("name", "title", imp);

      fObj->Browse(br);

      delete br;

      // no need to return object itself
      // TODO: make more exact check
      if (iter->NumElements() < 2) return nullptr;

      return iter;
   }

   /** Return TObject depending from kind of requested result */
   std::unique_ptr<RObject> GetObject(bool plain = false) override
   {

      if (plain)
         return std::make_unique<RTObjectHolder>(fObj);

      TObject *tobj = fObj->Clone();
      if (!tobj)
         return nullptr;
      if (tobj->IsA()->GetBaseClassOffset(TH1::Class()) == 0)
         static_cast<TH1 *>(tobj)->SetDirectory(nullptr);
      return std::make_unique<RUnique<TObject>>(tobj);
   }

   std::string ClassName() const { return fObj->ClassName(); }

};


// ==============================================================================================


void TMyBrowserImp::Add(TObject *obj, const char *name, Int_t)
{
   fIter->AddElement(std::make_shared<TObjectElement>(obj, name ? name : ""));
}


// ==============================================================================================


/** Create element for the browser */
std::unique_ptr<RBrowserItem> TObjectLevelIter::CreateBrowserItem()
{
   std::shared_ptr<TObjectElement> elem = std::dynamic_pointer_cast<TObjectElement>(fElements[fCounter]);

   auto item = std::make_unique<RBrowserTObjectItem>(elem->GetName(), -1);

   item->SetClassName(elem->ClassName());

   return item;
}


// ==============================================================================================


class RTObjectProvider : public RProvider {
protected:

   std::shared_ptr<RElement> DoBrowse(const TClass *cl, const void *object) const
   {
      if (cl && cl->InheritsFrom(TObject::Class())) {

         TObject *to = (TObject *) const_cast<TClass *>(cl)->DynamicCast(TObject::Class(), object, kTRUE);

         if (to) {
            printf("Doing browsing of obj %s %s\n", to->GetName(), to->ClassName());
            return std::make_shared<TObjectElement>(to);
         }
      }

      return nullptr;
   }

};

struct RTObjectProviderReg {
   std::shared_ptr<RTObjectProvider> provider;
   RTObjectProviderReg() { provider = std::make_shared<RTObjectProvider>(); RProvider::RegisterBrowse(nullptr, provider); }
   ~RTObjectProviderReg() { RProvider::Unregister(provider); }
} newRTObjectProviderReg;


