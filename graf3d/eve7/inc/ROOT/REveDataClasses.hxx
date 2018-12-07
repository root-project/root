// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT7_REveDataClasses
#define ROOT7_REveDataClasses

#include <ROOT/REveElement.hxx>

#include "TClass.h"

#include <functional>
#include <vector>
#include <iostream>

namespace ROOT {
namespace Experimental {

class REveDataItem;

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

   struct ItemInfo_t
   {
      void *fDataPtr{nullptr};
      REveDataItem *fItemPtr{nullptr};

      ItemInfo_t(void *dp, REveDataItem *di) : fDataPtr(dp), fItemPtr(di) {}
   };

   std::vector<ItemInfo_t> fItems;

   TString fFilterExpr;
   std::function<bool(void *)> fFilterFoo = [](void *) { return true; };

   REveDataCollection(const std::string& n = "REveDataCollection", const std::string& t = "");
   virtual ~REveDataCollection() {}

   TClass *GetItemClass() const { return fItemClass; }
   void SetItemClass(TClass *cls) { fItemClass = cls; }

   void ReserveItems(Int_t items_size) { fItems.reserve(items_size); }
   void AddItem(void *data_ptr, const std::string& n, const std::string& t);
   void ClearItems() { fItems.clear(); }

   void SetFilterExpr(const TString &filter);
   void ApplyFilter();

   Int_t GetNItems() const { return (Int_t)fItems.size(); }
   void *GetDataPtr(Int_t i) const { return fItems[i].fDataPtr; }
   REveDataItem *GetDataItem(Int_t i) const { return fItems[i].fItemPtr; }

   virtual Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset);

   virtual void SetCollectionColorRGB(UChar_t r, UChar_t g, UChar_t b);
   virtual void SetCollectionVisible(bool);
   virtual void ItemChanged(REveDataItem* item);

   void SetHandlerFunc (std::function<void (REveDataCollection*)> handler_func)
   {
      _handler_func = handler_func;
   }
   void SetHandlerFuncIds (std::function<void (REveDataCollection*, const Ids_t&)> handler_func)
   {
      _handler_func_ids= handler_func;
   }

   ClassDef(REveDataCollection, 0);
};

//==============================================================================

class REveDataItem : public REveElement
{
protected:
   Bool_t fFiltered{false};

public:
   REveDataItem(const std::string& n = "REveDataItem", const std::string& t = "");
   virtual ~REveDataItem() {}

   Bool_t GetFiltered() const { return fFiltered; }
   void   SetFiltered(Bool_t f);

   virtual void SetItemColorRGB(UChar_t r, UChar_t g, UChar_t b);
   virtual void SetItemRnrSelf(bool);

   Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset); // override;

   ClassDef(REveDataItem, 0);
};

//==============================================================================

class REveDataTable : public REveElement
{
protected:
   const REveDataCollection *fCollection{nullptr};

public:
   REveDataTable(const std::string& n = "REveDataTable", const std::string& t = "");
   virtual ~REveDataTable() {}

   void SetCollection(const REveDataCollection *col) { fCollection = col; }
   const REveDataCollection *GetCollection() const { return fCollection; }

   void PrintTable();
   virtual Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset);

   void AddNewColumn(const std::string& expr, const std::string& title, int prec = 2);

   ClassDef(REveDataTable, 0);
};

//==============================================================================

class REveDataColumn : public REveElement
{
public:
   enum FieldType_e { FT_Double = 0, FT_Bool, FT_String };

protected:
public:
   TString fExpression;
   FieldType_e fType; // can we auto detect this?
   Int_t fPrecision{2};

   std::string fTrue{"*"};
   std::string fFalse{" "};

   std::function<double(void *)> fDoubleFoo;
   std::function<bool(void *)> fBoolFoo;
   std::function<std::string(void *)> fStringFoo;

public:
   REveDataColumn(const std::string& n = "REveDataColumn", const std::string& t = "");
   virtual ~REveDataColumn() {}

   void SetExpressionAndType(const std::string &expr, FieldType_e type);
   void SetPrecision(Int_t prec);

   std::string EvalExpr(void *iptr);

   ClassDef(REveDataColumn, 0);
};


} // namespace Experimental
} // namespace ROOT

#endif
