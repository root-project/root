#include "ROOT/TEveDataClasses.hxx"

#include "TROOT.h"

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;


//==============================================================================
// TEveDataCollection
//==============================================================================

TEveDataCollection::TEveDataCollection(const char* n, const char* t) :
   TEveElementList(n, t)
{
   fChildClass = TEveDataItem::Class();
}

void TEveDataCollection::AddItem(void *data_ptr, const char* n, const char* t)
{
   auto el = new TEveDataItem(n, t);
   AddElement(el);
   fItems.push_back({data_ptr, el});
}

//------------------------------------------------------------------------------

void TEveDataCollection::SetFilterExpr(const TString& filter)
{
   static const TEveException eh("TEveDataCollection::SetFilterExpr ");

   if ( ! fItemClass) throw eh + "item class has to be set before the filter expression.";

   fFilterExpr = filter;

   TString s;
   s.Form("*((std::function<bool(%s*)>*)%p) = [](%s* p){%s &i=*p; return (%s); }",
          fItemClass->GetName(), &fFilterFoo, fItemClass->GetName(), fItemClass->GetName(),
          fFilterExpr.Data());

   printf("%s\n", s.Data());

   gROOT->ProcessLine(s.Data());
}

void TEveDataCollection::ApplyFilter()
{
   for (auto &ii : fItems)
   {
      bool res = fFilterFoo(ii.fDataPtr);

      printf("Item:%s -- filter result = %d\n", ii.fItemPtr->GetElementName(), res);

      ii.fItemPtr->SetFiltered( ! res );
   }
}


//==============================================================================
// TEveDataItem
//==============================================================================

TEveDataItem::TEveDataItem(const char* n, const char* t) :
   TEveElementList(n, t)
{
}


//==============================================================================
// TEveDataTable
//==============================================================================

TEveDataTable::TEveDataTable(const char* n, const char* t) :
   TEveElementList(n, t)
{
   fChildClass = TEveDataColumn::Class();
}


//==============================================================================
// TEveDataColumn
//==============================================================================

TEveDataColumn::TEveDataColumn(const char* n, const char* t) :
   TEveElementList(n, t)
{
}

void TEveDataColumn::SetExpressionAndType(const TString& expr, FieldType_e type)
{
   auto table = dynamic_cast<TEveDataTable*>(fMother);
   auto coll = table->GetCollection();
   auto icls = coll->GetItemClass();

   fExpression = expr;
   fType       = type;

   const char *rtyp;
   const void *fooptr;

   switch (fType)
   {
      case FT_Double: rtyp = "double";      fooptr = &fDoubleFoo; break;
      case FT_Bool:   rtyp = "bool";        fooptr = &fBoolFoo;   break;
      case FT_String: rtyp = "std::string"; fooptr = &fStringFoo; break;
   }

   TString s;
   s.Form("*((std::function<%s(%s*)>*)%p) = [](%s* p){%s &i=*p; return (%s); }",
          rtyp, icls->GetName(), fooptr, icls->GetName(), icls->GetName(),
          fExpression.Data());

   printf("%s\n", s.Data());

   gROOT->ProcessLine(s.Data());
}

void TEveDataColumn::SetPrecision(Int_t prec)
{
   fPrecision = prec;
}

std::string TEveDataColumn::EvalExpr(void *iptr)
{
   switch (fType)
   {
      case FT_Double:
      {
         TString ostr;
         ostr.Form("%.*f", fPrecision, fDoubleFoo(iptr));
         return ostr.Data();
      }
      case FT_Bool:
      {
         return fBoolFoo(iptr) ? fTrue : fFalse;
      }
      case FT_String:
      {
         return fStringFoo(iptr);
      }
   }
   return "XYZZ";
}
