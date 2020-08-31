// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT7_REveDataCollection
#define ROOT7_REveDataCollection

#include <ROOT/REveElement.hxx>

#include <functional>
#include <vector>
#include <iostream>

class TClass;

namespace ROOT {
namespace Experimental {

class REveDataItem;


//==============================================================================
// could be a nested class ???
class REveDataItem
{
   friend class REveDataCollection;

protected:
   void*    fDataPtr{nullptr};

   Bool_t   fRnrSelf{true};
   Color_t  fColor{0};
   Bool_t   fFiltered{false};


public:
   REveDataItem(void* d, Color_t c): fDataPtr(d), fColor(c){}

   Bool_t  GetRnrSelf() const { return fRnrSelf; }
   Color_t GetMainColor()   const { return fColor; }
   Bool_t  GetFiltered() const { return fFiltered; }


   void SetFiltered(Bool_t i) { fFiltered = i; }
   void SetMainColor(Color_t i) { fColor = i; }
   void SetRnrSelf(Bool_t i) { fRnrSelf = i; }
};

//==============================================================================

class REveDataCollection : public REveElement
{
public:
   typedef std::vector<int> Ids_t;

private:
   std::function<void (REveDataCollection*)>               _handler_func;
   std::function<void (REveDataCollection*, const Ids_t&)> _handler_func_ids;

public:
   static Color_t fgDefaultColor;

   TClass *fItemClass{nullptr}; // so far only really need class name

   std::vector<REveDataItem*> fItems;

   TString fFilterExpr;
   std::function<bool(void *)> fFilterFoo = [](void *) { return true; };

   REveDataCollection(const std::string& n = "REveDataCollection", const std::string& t = "");
   virtual ~REveDataCollection() {}

   Bool_t SingleRnrState() const override { return kTRUE; }
   Bool_t SetRnrState(Bool_t) override;

   void SetItemVisible(Int_t idx, Bool_t visible);
   void SetItemColorRGB(Int_t idx, UChar_t r, UChar_t g, UChar_t b);

   TClass *GetItemClass() const { return fItemClass; }
   void SetItemClass(TClass *cls) { fItemClass = cls; }

   void ReserveItems(Int_t items_size) { fItems.reserve(items_size); }
   void AddItem(void *data_ptr, const std::string& n, const std::string& t);
   void ClearItems() { fItems.clear(); }

   void SetFilterExpr(const TString &filter);
   void ApplyFilter();

   Int_t GetNItems() const { return (Int_t)fItems.size(); }
   void *GetDataPtr(Int_t i) const { return fItems[i]->fDataPtr; }
   //   const REveDataItem& RefDataItem(Int_t i) const { return fItems[i]; }
    const REveDataItem* GetDataItem(Int_t i) const { return fItems[i]; }

   void  StreamPublicMethods(nlohmann::json &cj);
   Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset) override;

   void SetMainColor(Color_t) override;
   virtual void ItemChanged(REveDataItem *item);
   virtual void ItemChanged(Int_t idx);

   void SetHandlerFunc (std::function<void (REveDataCollection*)> handler_func)
   {
      _handler_func = handler_func;
   }
   void SetHandlerFuncIds (std::function<void (REveDataCollection*, const Ids_t&)> handler_func)
   {
      _handler_func_ids= handler_func;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
