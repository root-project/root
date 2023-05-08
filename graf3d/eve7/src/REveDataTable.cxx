// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveDataTable.hxx>
#include <ROOT/REveDataCollection.hxx>
#include <ROOT/REveUtil.hxx>
#include <ROOT/RLogger.hxx>
#include "TClass.h"
#include "TROOT.h"

#include <sstream>

#include <nlohmann/json.hpp>

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
      // const REveDataItem &item = fCollection->RetDataItem(i);

      // !!!      printf("| %-20s |", item->GetCName());

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

         try {
            row[chld->GetCName()] = clmn->EvalExpr(data);
         }
         catch (const std::exception&) {
            R__LOG_ERROR(REveLog()) << "can't eval expr " << clmn->fExpression.Data();
            row[chld->GetCName()] = "err";
         }
      }
      jarr.push_back(row);
   }
   j["body"] = jarr;
   fCollection->StreamPublicMethods(j);
   j["fCollectionId"] = fCollection->GetElementId();
   return ret;
}

void REveDataTable::AddNewColumn(const std::string &expr, const std::string &title, int prec)
{
   auto c = new REveDataColumn(title);
   c->SetExpressionAndType(expr, REveDataColumn::FT_Double);
   c->SetPrecision(prec);
   gROOT->ProcessLine(c->GetFunctionExpressionString().c_str());

   if (c->hasValidExpression()) {
      AddElement(c);
      StampObjProps();
   }
}

//==============================================================================
// REveDataColumn
//==============================================================================

REveDataColumn::REveDataColumn(const std::string& n, const std::string& t) :
   REveElement(n, t)
{
}

//______________________________________________________________________________

void REveDataColumn::SetExpressionAndType(const std::string& expr, FieldType_e type, TClass* icls)
{
   fExpression = expr;
   fType       = type;
   fClassType  = icls;
}

//______________________________________________________________________________
void REveDataColumn::SetExpressionAndType(const std::string& expr, FieldType_e type)
{
   auto table = dynamic_cast<REveDataTable*>(fMother);
   auto coll = table->GetCollection();
   auto icls = coll->GetItemClass();
   SetExpressionAndType(expr, type, icls);
}

//______________________________________________________________________________
void REveDataColumn::SetPrecision(Int_t prec)
{
   fPrecision = prec;
}

//______________________________________________________________________________
std::string REveDataColumn::GetFunctionExpressionString() const
{
   const char *rtyp   = nullptr;
   const void *fooptr = nullptr;

   switch (fType)
   {
      case FT_Double: rtyp = "double";      fooptr = &fDoubleFoo; break;
      case FT_Bool:   rtyp = "bool";        fooptr = &fBoolFoo;   break;
      case FT_String: rtyp = "std::string"; fooptr = &fStringFoo; break;
   }

   std::stringstream s;
   s  << " *((std::function<" << rtyp << "(" << fClassType->GetName() << "*)>*)"
     << std::hex << std::showbase << (size_t)fooptr
     << ") = [](" << fClassType->GetName() << "* p){" << fClassType->GetName() << " &i=*p; return (" << fExpression.Data()
     << "); };";


  return s.str();
}

//______________________________________________________________________________
bool REveDataColumn::hasValidExpression() const
{
   return (fDoubleFoo || fBoolFoo || fStringFoo);
}

//______________________________________________________________________________
std::string REveDataColumn::EvalExpr(void *iptr) const
{
   if (!hasValidExpression())
      return "ErrFunc";

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
   return "Nn";
}
