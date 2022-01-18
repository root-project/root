/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RAttrBase.hxx>

#include <ROOT/RLogger.hxx>

#include <utility>

using namespace ROOT::Experimental;

RLogChannel &ROOT::Experimental::GPadLog()
{
   static RLogChannel sLog("ROOT.GPad");
   return sLog;
}

///////////////////////////////////////////////////////////////////////////////
/// Clear internal data

void RAttrBase::ClearData()
{
   if ((fKind == kOwnAttr) && fD.ownattr) {
      delete fD.ownattr;
      fD.ownattr = nullptr;
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Creates own attribute - only if no drawable and no parent are assigned

RAttrMap *RAttrBase::CreateOwnAttr()
{
   if (((fKind == kParent) && !fD.parent) || ((fKind == kDrawable) && !fD.drawable))
      fKind = kOwnAttr;

   if (fKind != kOwnAttr)
      return nullptr;

   if (!fD.ownattr)
      fD.ownattr = new RAttrMap();

   return fD.ownattr;
}

///////////////////////////////////////////////////////////////////////////////
/// Clear value if any with specified name

void RAttrBase::ClearValue(const std::string &name)
{
   if (auto access = AccessAttr(name))
       access.attr->Clear(access.fullname);
}

///////////////////////////////////////////////////////////////////////////////
/// Set `<NoValue>` for attribute. Ensure that value can not be configured via style - defaults will be used
/// Equivalent to css syntax { attrname:; }

void RAttrBase::SetNoValue(const std::string &name)
{
   if (auto access = AccessAttr(name))
       access.attr->AddNoValue(access.fullname);
}

///////////////////////////////////////////////////////////////////////////////
/// Move all fields into target object

void RAttrBase::MoveTo(RAttrBase &tgt)
{
   std::swap(fKind, tgt.fKind);
   std::swap(fD, tgt.fD);
   std::swap(fPrefix, tgt.fPrefix);
}
