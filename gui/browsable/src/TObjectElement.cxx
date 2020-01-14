/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RElement.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>
#include <ROOT/Browsable/RProvider.hxx>
#include <ROOT/Browsable/TObjectHolder.hxx>
#include <ROOT/Browsable/TObjectItem.hxx>

#include <ROOT/RLogger.hxx>

#include "TBrowser.h"
#include "TBrowserImp.h"
#include "TFolder.h"
#include "TList.h"
#include "TObjArray.h"
#include "TDirectory.h"

#include <sstream>

using namespace std::string_literals;

using namespace ROOT::Experimental::Browsable;

/** \class TObjectLevelIter
\ingroup rbrowser

Iterator over keys in TDirectory
*/


class TObjectLevelIter : public RLevelIter {

   std::vector<std::shared_ptr<RElement>> fElements;

   int fCounter{-1};

public:
   explicit TObjectLevelIter() = default;

   virtual ~TObjectLevelIter() = default;

   void AddElement(std::shared_ptr<RElement> &&elem)
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
   std::unique_ptr<RItem> CreateItem() override;

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override
   {
      return fElements[fCounter];
   }

};

// ===============================================================================================================


class TMyBrowserImp : public TBrowserImp {
   TObjectLevelIter &fIter;   ///<!  back-reference on iterator

public:

   TMyBrowserImp(TObjectLevelIter &iter) : TBrowserImp(nullptr), fIter(iter) {}
   virtual ~TMyBrowserImp() = default;

   void Add(TObject* obj, const char* name, Int_t) override;
};

// ===============================================================================================================


class TObjectElement : public RElement {
protected:
   std::unique_ptr<RHolder> fObject;
   TObject *fObj{nullptr};
   std::string fName;

public:
   TObjectElement(TObject *obj, const std::string &name = "") : fObj(obj), fName(name)
   {
      fObject = std::make_unique<TObjectHolder>(fObj);
      if (fName.empty())
         fName = fObj->GetName();
   }

   TObjectElement(std::unique_ptr<RHolder> &obj, const std::string &name = "")
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


///////////////////////////////////////////////////////////////
/// Create element for the browser

std::unique_ptr<RItem> TObjectLevelIter::CreateItem()
{
   std::shared_ptr<TObjectElement> elem = std::dynamic_pointer_cast<TObjectElement>(fElements[fCounter]);

   std::string clname = elem->ClassName();
   bool can_have_childs = (clname.find("TDirectory") == 0) || (clname.find("TTree") == 0) || (clname.find("TNtuple") == 0);

   auto item = std::make_unique<TObjectItem>(elem->GetName(), can_have_childs ? 1 : 0);

   item->SetClassName(elem->ClassName());

   item->SetIcon(RProvider::GetClassIcon(elem->ClassName()));

   return item;
}


// ==============================================================================================

class TFolderElement : public TObjectElement {

public:

   TFolderElement(std::unique_ptr<RHolder> &obj) : TObjectElement(obj) {}

   std::unique_ptr<RLevelIter> GetChildsIter() override;
};

class TCollectionElement : public TObjectElement {
public:

   TCollectionElement(std::unique_ptr<RHolder> &obj) : TObjectElement(obj) {}

   std::unique_ptr<RLevelIter> GetChildsIter() override;
};



class TCollectionIter : public RLevelIter {

   TIter  fIter;

public:
   explicit TCollectionIter(const TFolder *f) : RLevelIter(), fIter(f->GetListOfFolders()) {};

   explicit TCollectionIter(const TCollection *coll) : RLevelIter(), fIter(coll) {};

   virtual ~TCollectionIter() = default;

   bool Reset() override { fIter.Reset(); return true; }

   bool Next() override { return fIter.Next() != nullptr; }

   // use default implementation for now
   // bool Find(const std::string &name) override { return FindDirEntry(name); }

   bool HasItem() const override { return *fIter != nullptr; }

   std::string GetName() const override { return (*fIter)->GetName(); }

   int CanHaveChilds() const override { return 1; }

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
/// Provides iterator for generic TCollecion

std::unique_ptr<RLevelIter> TCollectionElement::GetChildsIter()
{
   auto coll = fObject->Get<TCollection>();
   if (coll)
      return std::make_unique<TCollectionIter>(coll);

   return TObjectElement::GetChildsIter();
}

// ==============================================================================================

class RTObjectProvider : public RProvider {

public:
   RTObjectProvider()
   {
      RegisterBrowse(TFolder::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TFolderElement>(object);
      });

      auto coll_labmda = [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TCollectionElement>(object);
      };

      RegisterBrowse(TList::Class(), coll_labmda);
      RegisterBrowse(TObjArray::Class(), coll_labmda);

      RegisterBrowse(nullptr, [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         if (object->CanCastTo<TFolder>())
            return std::make_shared<TFolderElement>(object);
         if (object->CanCastTo<TCollection>())
            return std::make_shared<TCollectionElement>(object);
         if (object->CanCastTo<TObject>() && !object->CanCastTo<TDirectory>()) // TODO: how to fix order of invoking
            return std::make_shared<TObjectElement>(object);
         return nullptr;
      });
   }

} newRTObjectProvider;
