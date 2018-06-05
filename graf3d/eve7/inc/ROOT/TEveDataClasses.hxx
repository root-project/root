#ifndef ROOT_TEveDataClasses_hxx
#define ROOT_TEveDataClasses_hxx

#include "ROOT/TEveElement.hxx"

#include "TClass.h"

#include <functional>
#include <vector>


namespace ROOT { namespace Experimental
{

class TEveDataItem;

//==============================================================================

class TEveDataCollection : public TEveElementList
{
protected:
   TClass       *fItemClass = 0; // so far only really need class name

   struct ItemInfo_t
   {
      void         *fDataPtr;
      TEveDataItem *fItemPtr;

      ItemInfo_t(void *dp, TEveDataItem *di) : fDataPtr(dp), fItemPtr(di) {}
   };

   std::vector<ItemInfo_t> fItems;

   TString                 fFilterExpr;

public:
   TEveDataCollection(const char* n="TEveDataCollection", const char* t="");
   virtual ~TEveDataCollection() {}

   void ReserveItems(Int_t items_size) { fItems.reserve(items_size); }
   void AddItem(void *data_ptr, const char* n, const char* t);

   void SetFilterExpr(const TString& filter) { fFilterExpr = filter; }
   void ApplyFilter();

   ClassDef(TEveDataCollection, 0);
};

//==============================================================================

class TEveDataItem : public TEveElementList // XXXXX
{
protected:
   Bool_t    fFiltered = false;

public:
   TEveDataItem(const char* n="TEveDataItem", const char* t="");
   virtual ~TEveDataItem() {}

   Bool_t GetFiltered() const   { return fFiltered; }
   void   SetFiltered(Bool_t f) { if (f != fFiltered) { fFiltered = f; /* stamp; */ } };

   ClassDef(TEveDataItem, 0);
};

//==============================================================================

class TEveDataTable : public TEveElementList // XXXX
{
protected:
   TEveDataCollection *fCollection;

public:
   TEveDataTable(const char* n="TEveDataTable", const char* t="");
   virtual ~TEveDataTable() {}

   ClassDef(TEveDataTable, 0);
};

//==============================================================================

class TEveDataColumn : public TEveElementList // XXXX
{
protected:
   TString  fType;       // probably string, int, float, double; should be enum?
   TString  fExpression;
   Int_t    fPrecision;

public:
   TEveDataColumn(const char* n="TEveDataColumn", const char* t="");
   virtual ~TEveDataColumn() {}

   ClassDef(TEveDataColumn, 0);
};

}}

#endif
