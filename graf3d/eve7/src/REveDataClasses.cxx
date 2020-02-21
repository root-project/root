// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveDataClasses.hxx>
#include <ROOT/REveUtil.hxx>

#include "TROOT.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TColor.h"
#include "TClass.h"
#include "TBaseClass.h"

#include "json.hpp"
#include <sstream>


using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;


Color_t  REveDataCollection::fgDefaultColor  = kBlue;

//==============================================================================
// REveDataCollection
//==============================================================================

REveDataCollection::REveDataCollection(const std::string& n, const std::string& t) :
   REveElement(n, t)
{
   fChildClass = TClass::GetClass<REveDataItem>();

   SetupDefaultColorAndTransparency(fgDefaultColor, true, true);

   _handler_func = 0;
   _handler_func_ids = 0;
}

void REveDataCollection::AddItem(void *data_ptr, const std::string& n, const std::string& t)
{
   auto el = new REveDataItem(n, t);
   AddElement(el);
   el->SetMainColor(GetMainColor());
   fItems.emplace_back(data_ptr, el);
}

//------------------------------------------------------------------------------

void REveDataCollection::SetFilterExpr(const TString& filter)
{
   static const REveException eh("REveDataCollection::SetFilterExpr ");

   if (!fItemClass) throw eh + "item class has to be set before the filter expression.";

   fFilterExpr = filter;

   std::stringstream s;
   s << "*((std::function<bool(" << fItemClass->GetName() << "*)>*)" << std::hex << std::showbase << (size_t)&fFilterFoo
     << ") = [](" << fItemClass->GetName() << "* p){" << fItemClass->GetName() << " &i=*p; return ("
     << fFilterExpr.Data() << "); }";

   // printf("%s\n", s.Data());
   try {
      gROOT->ProcessLine(s.str().c_str());
      // AMT I don't know why ApplyFilter call is separated
      ApplyFilter();
   }
   catch (const std::exception &exc)
   {
      std::cerr << "EveDataCollection::SetFilterExpr" << exc.what();
   }

}

void REveDataCollection::ApplyFilter()
{
   Ids_t ids;
   int idx = 0;
   for (auto &ii : fItems)
   {
      bool res = fFilterFoo(ii.fDataPtr);

      // printf("Item:%s -- filter result = %d\n", ii.fItemPtr->GetElementName(), res);

      ii.fItemPtr->SetFiltered( ! res );

      ids.push_back(idx++);
   }
   StampObjProps();
   if (_handler_func_ids) _handler_func_ids( this , ids);
}

//______________________________________________________________________________

Int_t REveDataCollection::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   struct PubMethods
   {
      void FillJSON(TClass* c, nlohmann::json & arr)
      {
         TString  ctor = c->GetName(), dtor = "~";
         {
            int i = ctor.Last(':');
            if (i != kNPOS)
            {
               ctor.Replace(0, i + 1, "");
            }
            dtor += ctor;
         }

         TMethod *meth;
         TIter    next(c->GetListOfMethods());
         while ((meth = (TMethod*) next()))
         {
            // Filter out ctor, dtor, some ROOT stuff.
            {
               TString m(meth->GetName());
               if (m == ctor || m == dtor ||
                   m == "Class" || m == "Class_Name" || m == "Class_Version" || m == "Dictionary" || m == "IsA" ||
                   m == "DeclFileName" || m == "ImplFileName" || m == "DeclFileLine" || m == "ImplFileLine" ||
                   m == "Streamer" || m == "StreamerNVirtual" || m == "ShowMembers" ||
                   m == "CheckTObjectHashConsistency")
               {
                  continue;
               }
            }

            TString     ms;
            TMethodArg *ma;
            TIter       next_ma(meth->GetListOfMethodArgs());
            while ((ma = (TMethodArg*) next_ma()))
            {
               if ( ! ms.IsNull()) ms += ", ";

               ms += ma->GetTypeName();
               ms += " ";
               ms += ma->GetName();
            }
            char* entry = Form("i.%s(%s)",meth->GetName(),ms.Data());
            nlohmann::json jm ;
            jm["f"] = entry;
            jm["r"] = meth->GetReturnTypeName();
            jm["c"] = c->GetName();
            arr.push_back(jm);
         }
         {
            TBaseClass *base;
            TIter       blnext(c->GetListOfBases());
            while ((base = (TBaseClass*) blnext()))
            {
               FillJSON(base->GetClassPointer(), arr);
            }
         }
      }
   };

   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);
   j["fFilterExpr"] = fFilterExpr.Data();
   j["fPublicFunctions"]  = nlohmann::json::array();
   PubMethods pm;
   pm.FillJSON(fItemClass, j["fPublicFunctions"]);

   return ret;
}

//______________________________________________________________________________

void REveDataCollection::SetMainColor(Color_t newv)
{
   int idx = 0;
   Ids_t ids;
   for (auto & chld : fChildren)
   {
      chld->SetMainColor(newv);
      ids.push_back(idx);
      idx++;
   }

   REveElement::SetMainColor(newv);
   // printf("REveDataCollection::SetCollectionColorRGB color ched to %d ->%d\n", oldv, GetMainColor());
    if ( _handler_func_ids) _handler_func_ids( this , ids);
}

//______________________________________________________________________________

Bool_t REveDataCollection::SetRnrState(Bool_t iRnrSelf)
{
   Bool_t ret = REveElement::SetRnrState(iRnrSelf);

   Ids_t ids;

   for (int i = 0; i < GetNItems(); ++i ) {
      ids.push_back(i);
      GetDataItem(i)->SetRnrSelf(fRnrSelf);
   }

   _handler_func_ids( this , ids);

   return ret;
}



//______________________________________________________________________________

void REveDataCollection::ItemChanged(REveDataItem* iItem)
{
   int idx = 0;
   Ids_t ids;
   for (auto & chld : fChildren)
   {
      if (chld == iItem) {
         ids.push_back(idx);
         _handler_func_ids( this , ids);
         return;
      }
      idx++;
   }
}

//==============================================================================
// REveDataItem
//==============================================================================

REveDataItem::REveDataItem(const std::string& n, const std::string& t) :
   REveElement(n, t)
{
   SetupDefaultColorAndTransparency(kMagenta, true, true);
}

Int_t REveDataItem::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);
   j["fFiltered"] = fFiltered;
   return ret;
}

void REveDataItem::SetItemColorRGB(UChar_t r, UChar_t g, UChar_t b)
{
   Color_t color = TColor::GetColor(r, g, b);
   REveElement::SetMainColor(color);
   REveDataCollection* c = dynamic_cast<REveDataCollection*>(fMother);
   c->ItemChanged(this);
}

Bool_t REveDataItem::SetRnrSelf(Bool_t iRnrSelf)
{
   Bool_t r = REveElement::SetRnrSelf(iRnrSelf);
   REveDataCollection* c = dynamic_cast<REveDataCollection*>(fMother);
   c->ItemChanged(this);
   return r;
}

void REveDataItem::SetFiltered(bool f)
{
  if (f != fFiltered)
  {
     fFiltered = f;
     StampObjProps();
  }
}

void REveDataItem::FillImpliedSelectedSet(Set_t &impSelSet)
{
   for (auto &n : fNieces)
   {
      impSelSet.insert(n);
      n->FillImpliedSelectedSet(impSelSet);

      if (gDebug > 1)
      {
         printf("REveDataItem::FillImpliedSelectedSet added niece '%s' [%s]\n",
                n->GetCName(), n->IsA()->GetName());
      }
   }
}

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
   auto c = new REX::REveDataColumn(title);
   AddElement(c);

   c->SetExpressionAndType(expr, REX::REveDataColumn::FT_Double);
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
