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
public:
   TClass       *fItemClass = 0; // so far only really need class name

   struct ItemInfo_t
   {
      void         *fDataPtr;
      TEveDataItem *fItemPtr;

      ItemInfo_t(void *dp, TEveDataItem *di) : fDataPtr(dp), fItemPtr(di) {}
   };

   std::vector<ItemInfo_t> fItems;

   TString                    fFilterExpr;
   std::function<bool(void*)> fFilterFoo = [](void*){ return true; };

public:
   TEveDataCollection(const char* n="TEveDataCollection", const char* t="");
   virtual ~TEveDataCollection() {}

   TClass* GetItemClass()      const { return fItemClass; }
   void    SetItemClass(TClass* cls) { fItemClass = cls;  }

   void ReserveItems(Int_t items_size) { fItems.reserve(items_size); }
   void AddItem(void *data_ptr, const char* n, const char* t);

   void SetFilterExpr(const TString& filter);
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

   void SetCollection(TEveDataCollection *col) { fCollection = col;  }
   TEveDataCollection* GetCollection()   const { return fCollection; }

   ClassDef(TEveDataTable, 0);
};

//==============================================================================

class TEveDataColumn : public TEveElementList // XXXX
{
public:
   enum FieldType_e { FT_Double = 0, FT_Bool, FT_String };

protected:
public:
   TString      fExpression;
   FieldType_e  fType;       // can we auto detect this?
   Int_t        fPrecision = 2;

   std::string  fTrue  = "*";
   std::string  fFalse = " ";

   std::function<double(void*)>      fDoubleFoo;
   std::function<bool(void*)>        fBoolFoo;
   std::function<std::string(void*)> fStringFoo;

public:
   TEveDataColumn(const char* n="TEveDataColumn", const char* t="");
   virtual ~TEveDataColumn() {}

   void SetExpressionAndType(const TString& expr, FieldType_e type);
   void SetPrecision(Int_t prec);

   std::string EvalExpr(void *iptr);

   ClassDef(TEveDataColumn, 0);
};

}}

#endif
