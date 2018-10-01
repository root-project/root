#ifndef ROOT_REveDataClasses_hxx
#define ROOT_REveDataClasses_hxx

#include "ROOT/REveElement.hxx"

#include "TClass.h"

#include <functional>
#include <vector>
#include <iostream>

namespace ROOT {
namespace Experimental {

class REveDataItem;

//==============================================================================

class REveDataCollection : public REveElementList {
protected:
public:
   TClass *fItemClass{nullptr}; // so far only really need class name

   struct ItemInfo_t {
      void *fDataPtr;
      REveDataItem *fItemPtr;

      ItemInfo_t(void *dp, REveDataItem *di) : fDataPtr(dp), fItemPtr(di) {}
   };

   std::vector<ItemInfo_t> fItems;

   TString fFilterExpr;
   std::function<bool(void *)> fFilterFoo = [](void *) { return true; };

public:
   REveDataCollection(const char *n = "REveDataCollection", const char *t = "");
   virtual ~REveDataCollection() {}

   TClass *GetItemClass() const { return fItemClass; }
   void SetItemClass(TClass *cls) { fItemClass = cls; }

   void ReserveItems(Int_t items_size) { fItems.reserve(items_size); }
   void AddItem(void *data_ptr, const char *n, const char *t);

   void SetFilterExpr(const TString &filter);
   void ApplyFilter();

   Int_t GetNItems() const { return (Int_t)fItems.size(); }
   void *GetDataPtr(Int_t i) const { return fItems[i].fDataPtr; }
   REveDataItem *GetDataItem(Int_t i) const { return fItems[i].fItemPtr; }

   virtual Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset);

   ClassDef(REveDataCollection, 0);
};

//==============================================================================

class REveDataItem : public REveElementList {
protected:
   Bool_t fFiltered{false};

public:
   REveDataItem(const char *n = "REveDataItem", const char *t = "");
   virtual ~REveDataItem() {}

   Bool_t GetFiltered() const { return fFiltered; }
   void SetFiltered(Bool_t f)
   {
      if (f != fFiltered) {
         fFiltered = f; /* stamp; */
      }
   };

   virtual Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset);
   ClassDef(REveDataItem, 0);
};

//==============================================================================

class REveDataTable : public REveElementList // XXXX
{
protected:
   REveDataCollection *fCollection{nullptr};

public:
   REveDataTable(const char *n = "REveDataTable", const char *t = "");
   virtual ~REveDataTable() {}

   void SetCollection(REveDataCollection *col) { fCollection = col; }
   REveDataCollection *GetCollection() const { return fCollection; }

   void PrintTable();
   virtual Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset);

   void AddNewColumn(const char *expr, const char *title, int prec = 2);

   ClassDef(REveDataTable, 0);
};

//==============================================================================

class REveDataColumn : public REveElementList // XXXX
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
   REveDataColumn(const char *n = "REveDataColumn", const char *t = "");
   virtual ~REveDataColumn() {}

   void SetExpressionAndType(const TString &expr, FieldType_e type);
   void SetPrecision(Int_t prec);

   std::string EvalExpr(void *iptr);

   ClassDef(REveDataColumn, 0);
};

} // namespace Experimental
} // namespace ROOT

#endif
