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

RDrawableProvider::Map_t &RDrawableProvider::GetV6Map()
{
   static RDrawableProvider::Map_t sMap;
   return sMap;
}

RDrawableProvider::Map_t &RDrawableProvider::GetV7Map()
{
   static RDrawableProvider::Map_t sMap;
   return sMap;
}

void RDrawableProvider::RegisterV6(const TClass *cl, std::shared_ptr<RDrawableProvider> provider)
{
    auto &bmap = GetV6Map();

    if (cl && (bmap.find(cl) != bmap.end()))
       R__ERROR_HERE("Browserv7") << "Drawable handler for class " << cl->GetName() << " already exists";

    bmap.emplace(cl, provider);
}

void RDrawableProvider::RegisterV7(const TClass *cl, std::shared_ptr<RDrawableProvider> provider)
{
    auto &bmap = GetV7Map();

    if (cl && (bmap.find(cl) != bmap.end()))
       R__ERROR_HERE("Browserv7") << "Drawable handler for class " << cl->GetName() << " already exists";

    bmap.emplace(cl, provider);
}

//////////////////////////////////////////////////////////////////////////////////
// remove provider from all registered lists

void RDrawableProvider::Unregister(std::shared_ptr<RDrawableProvider> provider)
{
   auto &map6 = GetV6Map();
   for (auto iter6 = map6.begin(); iter6 != map6.end();) {
      if (iter6->second == provider)
         iter6 = map6.erase(iter6);
      else
         iter6++;
   }

   auto &map7 = GetV7Map();
   for (auto iter7 = map7.begin(); iter7 != map7.end();) {
      if (iter7->second == provider)
         iter7 = map7.erase(iter7);
      else
         iter7++;
   }
}

bool RDrawableProvider::DrawV6(TPad *subpad, std::unique_ptr<Browsable::RObject> &obj, const std::string &opt)
{
   auto &map6 = GetV6Map();
   auto iter6 = map6.find(obj->GetClass());

   if (iter6 != map6.end()) {
      if (iter6->second->DoDrawV6(subpad, obj, opt))
         return true;
   }

   for (auto &pair : map6)
      if ((pair.first == obj->GetClass()) || !pair.first)
         if (pair.second->DoDrawV6(subpad, obj, opt))
            return true;

   return false;
}

bool RDrawableProvider::DrawV7(std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RObject> &obj, const std::string &opt)
{
   auto &map7 = GetV7Map();
   auto iter7 = map7.find(obj->GetClass());

   if (iter7 != map7.end()) {
      if (iter7->second->DoDrawV7(subpad, obj, opt))
         return true;
   }

   for (auto &pair : map7)
      if ((pair.first == obj->GetClass()) || !pair.first)
         if (pair.second->DoDrawV7(subpad, obj, opt))
            return true;

   return false;
}
