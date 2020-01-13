/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDrawableProvider.hxx"

#include "ROOT/RLogger.hxx"

#include "TBaseClass.h"
#include "TList.h"
#include "TSystem.h"

using namespace ROOT::Experimental;

//////////////////////////////////////////////////////////////////////////////////
// Returns map of registered drawing functions for v6 canvas

RDrawableProvider::MapV6_t &RDrawableProvider::GetV6Map()
{
   static RDrawableProvider::MapV6_t sMap;
   return sMap;
}

//////////////////////////////////////////////////////////////////////////////////
// Returns map of registered drawing functions for v7 canvas

RDrawableProvider::MapV7_t &RDrawableProvider::GetV7Map()
{
   static RDrawableProvider::MapV7_t sMap;
   return sMap;
}

//////////////////////////////////////////////////////////////////////////////////
// Register drawing function for v6 canvas

void RDrawableProvider::RegisterV6(const TClass *cl, FuncV6_t func)
{
    auto &bmap = GetV6Map();

    if (cl && (bmap.find(cl) != bmap.end()))
       R__ERROR_HERE("Browserv7") << "Drawable handler for class " << cl->GetName() << " already exists";

    bmap.emplace(cl, StructV6{this, func});
}

//////////////////////////////////////////////////////////////////////////////////
// Register drawing function for v7 canvas

void RDrawableProvider::RegisterV7(const TClass *cl, FuncV7_t func)
{
    auto &bmap = GetV7Map();

    if (cl && (bmap.find(cl) != bmap.end()))
       R__ERROR_HERE("Browserv7") << "Drawable handler for class " << cl->GetName() << " already exists";

    bmap.emplace(cl, StructV7{this, func});
}

//////////////////////////////////////////////////////////////////////////////////
// remove provider from all registered lists

RDrawableProvider::~RDrawableProvider()
{
   // TODO: cleanup itself from global list

   auto &map6 = GetV6Map();
   for (auto iter6 = map6.begin(); iter6 != map6.end();) {
      if (iter6->second.provider == this)
         iter6 = map6.erase(iter6);
      else
         iter6++;
   }

   auto &map7 = GetV7Map();
   for (auto iter7 = map7.begin(); iter7 != map7.end();) {
      if (iter7->second.provider == this)
         iter7 = map7.erase(iter7);
      else
         iter7++;
   }
}

/////////////////////////////////////////////////////////////////////////////////
/// Invoke drawing of object on TCanvas sub-pad
/// All existing providers are checked, first checked are class matches (including direct parents)

bool RDrawableProvider::DrawV6(TVirtualPad *subpad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt)
{
   if (!obj || !obj->GetClass())
      return false;

   auto &map6 = GetV6Map();

   TClass *cl = const_cast<TClass *>(obj->GetClass());
   while (cl) {
      auto iter6 = map6.find(cl);

      if (iter6 != map6.end()) {
         if (iter6->second.func(subpad, obj, opt))
            return true;
      }

      auto bases = cl->GetListOfBases();

      cl = bases && (bases->GetSize() > 0) ? dynamic_cast<TBaseClass *>(bases->First())->GetClassPointer() : nullptr;
   }

   for (auto &pair : map6)
      if ((pair.first == obj->GetClass()) || !pair.first)
         if (pair.second.func(subpad, obj, opt))
            return true;

   // try to load necessary library and repeat action again
   // TODO: need factory methods for that

   if (obj->GetClass()->InheritsFrom("TLeaf"))
      gSystem->Load("libROOTTreeDrawProvider");
   else if (obj->GetClass()->InheritsFrom(TObject::Class()))
      gSystem->Load("libROOTObjectDrawProvider");
   else
      return false;

   cl = const_cast<TClass *>(obj->GetClass());
   while (cl) {
      auto iter6 = map6.find(cl);

      if (iter6 != map6.end()) {
         if (iter6->second.func(subpad, obj, opt))
            return true;
      }

      auto bases = cl->GetListOfBases();

      cl = bases && (bases->GetSize() > 0) ? dynamic_cast<TBaseClass *>(bases->First())->GetClassPointer() : nullptr;
   }

   for (auto &pair : map6)
      if ((pair.first == obj->GetClass()) || !pair.first)
         if (pair.second.func(subpad, obj, opt))
            return true;

   return false;
}

/////////////////////////////////////////////////////////////////////////////////
/// Invoke drawing of object on RCanvas sub-pad
/// All existing providers are checked, first checked are class matches (including direct parents)

bool RDrawableProvider::DrawV7(std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt)
{
   if (!obj || !obj->GetClass())
      return false;

   auto &map7 = GetV7Map();

   TClass *cl = const_cast<TClass *>(obj->GetClass());
   while (cl) {
      auto iter7 = map7.find(cl);

      if (iter7 != map7.end()) {
         if (iter7->second.func(subpad, obj, opt))
            return true;
      }

      auto bases = cl->GetListOfBases();

      cl = bases && (bases->GetSize() > 0) ? dynamic_cast<TBaseClass *>(bases->First())->GetClassPointer() : nullptr;
   }

   for (auto &pair : map7)
      if ((pair.first == obj->GetClass()) || !pair.first)
         if (pair.second.func(subpad, obj, opt))
            return true;

   // try to load necessary library and repeat action again
   // TODO: need factory methods for that

   if (obj->GetClass()->InheritsFrom("TLeaf"))
      gSystem->Load("libROOTTreeDrawProvider");
   else if (obj->GetClass()->InheritsFrom(TObject::Class()))
      gSystem->Load("libROOTObjectDrawProvider");
   else if (obj->GetClass()->InheritsFrom("ROOT::Experimental::RH1D") || obj->GetClass()->InheritsFrom("ROOT::Experimental::RH2D") || obj->GetClass()->InheritsFrom("ROOT::Experimental::RH2D"))
      gSystem->Load("libROOTHistDrawProvider");
   else
      return false;

   cl = const_cast<TClass *>(obj->GetClass());
   while (cl) {
      auto iter7 = map7.find(cl);

      if (iter7 != map7.end()) {
         if (iter7->second.func(subpad, obj, opt))
            return true;
      }

      auto bases = cl->GetListOfBases();

      cl = bases && (bases->GetSize() > 0) ? dynamic_cast<TBaseClass *>(bases->First())->GetClassPointer() : nullptr;
   }

   for (auto &pair : map7)
      if ((pair.first == obj->GetClass()) || !pair.first)
         if (pair.second.func(subpad, obj, opt))
            return true;

   return false;
}
