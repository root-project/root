#include <ROOT/REveDataTable.hxx>
#include <ROOT/REveDataCollection.hxx>
#include <TClass.h>
#include <json.hpp>
#include <sstream>
#include <TROOT.h>

using namespace ROOT::Experimental;

//==============================================================================
// REveDataTable
//==============================================================================

REveDataTable::REveDataTable(const std::string& n, const std::string& t) :
   REveElement(n, t)
{
   fChildClass = TClass::GetClass<REveDataColumn>();
}

void REveDataTable::PrintTable()
{
   Int_t Nit = fCollection->GetNItems();

   for (Int_t i = 0; i< Nit; ++i)
   {
      void         *data = fCollection->GetDataPtr(i);
      REveDataItem *item = fCollection->GetDataItem(i);

      printf("| %-20s |", item->GetCName());

      for (auto & chld : fChildren)
      {
         auto clmn = dynamic_cast<REveDataColumn*>(chld);

         printf(" %10s |", clmn->EvalExpr(data).c_str());
      }
      printf("\n");
   }
}

Int_t REveDataTable::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   int ret = REveElement::WriteCoreJson(j, rnr_offset);
   Int_t Nit = fCollection->GetNItems();

   nlohmann::json jarr = nlohmann::json::array();

   for (Int_t i = 0; i < Nit; ++i) {
      void *data = fCollection->GetDataPtr(i);
      nlohmann::json row;
      for (auto &chld : fChildren) {
         auto clmn = dynamic_cast<REveDataColumn *>(chld);
         row[chld->GetCName()] = clmn->EvalExpr(data);
      }
      jarr.push_back(row);
   }
   j["body"] = jarr;
   j["fCollectionId"] = fCollection->GetElementId();
   return ret;
}

void REveDataTable::AddNewColumn(const std::string& expr, const std::string& title, int prec)
{
   auto c = new REveDataColumn(title);
   AddElement(c);

   c->SetExpressionAndType(expr, REveDataColumn::FT_Double);
   c->SetPrecision(prec);

   StampObjProps();
}

//==============================================================================
// REveDataColumn
//==============================================================================

REveDataColumn::REveDataColumn(const std::string& n, const std::string& t) :
   REveElement(n, t)
{
}

void REveDataColumn::SetExpressionAndType(const std::string& expr, FieldType_e type)
{
   auto table = dynamic_cast<REveDataTable*>(fMother);
   auto coll = table->GetCollection();
   auto icls = coll->GetItemClass();

   fExpression = expr;
   fType       = type;

   const char *rtyp   = nullptr;
   const void *fooptr = nullptr;

   switch (fType)
   {
      case FT_Double: rtyp = "double";      fooptr = &fDoubleFoo; break;
      case FT_Bool:   rtyp = "bool";        fooptr = &fBoolFoo;   break;
      case FT_String: rtyp = "std::string"; fooptr = &fStringFoo; break;
   }

   std::stringstream s;
   s << "*((std::function<" << rtyp << "(" << icls->GetName() << "*)>*)" << std::hex << std::showbase << (size_t)fooptr
     << ") = [](" << icls->GetName() << "* p){" << icls->GetName() << " &i=*p; return (" << fExpression.Data()
     << "); }";

   // printf("%s\n", s.str().c_str());
   try {
      gROOT->ProcessLine(s.str().c_str());
   }
   catch (const std::exception &exc)
   {
      std::cerr << "REveDataColumn::SetExpressionAndType" << exc.what();
   }
}

void REveDataColumn::SetPrecision(Int_t prec)
{
   fPrecision = prec;
}

std::string REveDataColumn::EvalExpr(void *iptr)
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
   return "XYZ";
}
