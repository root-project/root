/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDrawableProvider.hxx"

#include "ROOT/RLogger.hxx"

using namespace ROOT::Experimental;

RDrawableProvider::MapV6_t &RDrawableProvider::GetV6Map()
{
   static RDrawableProvider::MapV6_t sMap;
   return sMap;
}

RDrawableProvider::MapV7_t &RDrawableProvider::GetV7Map()
{
   static RDrawableProvider::MapV7_t sMap;
   return sMap;
}

void RDrawableProvider::RegisterV6(const TClass *cl, FuncV6_t func)
{
    auto &bmap = GetV6Map();

    if (cl && (bmap.find(cl) != bmap.end()))
       R__ERROR_HERE("Browserv7") << "Drawable handler for class " << cl->GetName() << " already exists";

    bmap.emplace(cl, StructV6{this, func});
}

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

bool RDrawableProvider::DrawV6(TVirtualPad *subpad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt)
{
   auto &map6 = GetV6Map();
   auto iter6 = map6.find(obj->GetClass());

   if (iter6 != map6.end()) {
      if (iter6->second.func(subpad, obj, opt))
         return true;
   }

   for (auto &pair : map6)
      if ((pair.first == obj->GetClass()) || !pair.first)
         if (pair.second.func(subpad, obj, opt))
            return true;

   return false;
}

bool RDrawableProvider::DrawV7(std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt)
{
   auto &map7 = GetV7Map();
   auto iter7 = map7.find(obj->GetClass());

   if (iter7 != map7.end()) {
      if (iter7->second.func(subpad, obj, opt))
         return true;
   }

   for (auto &pair : map7)
      if ((pair.first == obj->GetClass()) || !pair.first)
         if (pair.second.func(subpad, obj, opt))
            return true;

   return false;
}
