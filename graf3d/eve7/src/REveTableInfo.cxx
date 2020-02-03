// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TClass.h"
#include "TBaseClass.h"
#include "TROOT.h"
#include "TInterpreter.h"
#include "TMethod.h"
#include "TMethodArg.h"

#include <ROOT/REveTableInfo.hxx>
#include <ROOT/REveDataClasses.hxx>
#include <ROOT/REveManager.hxx>

#include "json.hpp"

using namespace ROOT::Experimental;

REveTableViewInfo::REveTableViewInfo(const std::string &name, const std::string &title) : REveElement(name, title), fConfigChanged(false)
{
}

void REveTableViewInfo::SetDisplayedCollection(ElementId_t collectionId)
{
   fDisplayedCollection = collectionId;

   fConfigChanged = true;
   for (auto &it : fDelegates)
      it();

   fConfigChanged = false;
   StampObjProps();
}

void REveTableViewInfo::AddNewColumnToCurrentCollection(const std::string& expr, const std::string& title, int prec)
{
   if (!fDisplayedCollection) return;

   REveDataCollection* col = dynamic_cast<REveDataCollection*>(gEve->FindElementById(fDisplayedCollection));
   if (!col) {
      printf("REveTableViewInfo::AddNewColumnToCurrentCollection error: collection not found\n");
      return;
   }

   const char *rtyp   = "void";
   auto icls = col->GetItemClass();
   std::function<void(void *)> fooptr;
   std::stringstream s;
   s << "*((std::function<" << rtyp << "(" << icls->GetName() << "*)>*)" << std::hex << std::showbase << (size_t)(&fooptr)
     << ") = [](" << icls->GetName() << "* p){" << icls->GetName() << " &i=*p; return (" << expr
     << "); }";

   int err;
   gROOT->ProcessLine(s.str().c_str(), &err);
   if (err != TInterpreter::kNoError)
   {
      std::cout << "REveTableViewInfo::AddNewColumnToCurrentCollection failed." <<  std::endl;
      return;
   }

   fConfigChanged = true;
   table(col->GetItemClass()->GetName()).column(title, prec, expr);

   for (auto &it : fDelegates)
      it();

   fConfigChanged = false;

   StampObjProps();
}


////////////////////////////////////////////////////////////////////////////////
/// Find column definitions for given class name.
//  Look for definition also in base classes
REveTableHandle::Entries_t& REveTableViewInfo::RefTableEntries(std::string cname)
{
   struct SimpleMethodFill
   {
      void fill(REveTableHandle::Entries_t& entries, TClass* c)
      {
         TMethod *meth;
         TIter    next(c->GetListOfMethods());
         while ((meth = (TMethod*) next())) {
            // take only methods without arguments
            if (!meth->GetListOfMethodArgs()->First())
            {
               std::string mn = meth->GetName();
               std::string exp = "i." + mn + "()";

               // ?? AMT is thre a better way to get detect numeric types
               std::string r  = meth->GetReturnTypeName();
               if ( r == "int"    || r == "Int_t" ||
                    r == "uint" || r == "UInt_t")
               {
                  entries.push_back(REveTableEntry(mn, 0, exp));
               }
               else if( r == "double" || r == "Double_t" ||
                        r == "float"  || r == "Float_t" )
               {
                  entries.push_back(REveTableEntry(mn, 0, exp));
               }
            }
         }



         // look in the base classes
         TBaseClass *base;
         TIter       blnext(c->GetListOfBases());
         while ((base = (TBaseClass*) blnext()))
         {
            fill(entries, base->GetClassPointer());
         }

      }
   };

   auto search = fSpecs.find(cname);
   if (search != fSpecs.end())
   {
      return search->second;
   }
   else {
      TClass* c = TClass::GetClass(cname.c_str());
      TBaseClass *base;
      TIter       blnext(c->GetListOfBases());
      while ((base = (TBaseClass*) blnext()))
      {
         auto bs = fSpecs.find(base->GetName());
         if (bs != fSpecs.end())
         {
            return bs->second;
         }
      }
   }

   // create new entry if not existing
   SimpleMethodFill methodFinder;
   methodFinder.fill(fSpecs[cname], TClass::GetClass(cname.c_str()));
   return fSpecs[cname];
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t REveTableViewInfo::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   auto ret = REveElement::WriteCoreJson(j, rnr_offset);
   j["fDisplayedCollection"] = fDisplayedCollection;
   return ret;
}


