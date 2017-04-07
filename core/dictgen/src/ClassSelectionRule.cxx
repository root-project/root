// @(#)root/core/utils:$Id: ClassSelectionRule.cxx 41697 2011-11-01 21:03:41Z pcanal $
// Author: Velislava Spasova September 2010

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ClassSelection                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "ClassSelectionRule.h"
#include <iostream>

void ClassSelectionRule::AddFieldSelectionRule(const VariableSelectionRule& field)
{
  fFieldSelectionRules.emplace_back(field);
}

bool ClassSelectionRule::HasFieldSelectionRules() const
{
  return !fFieldSelectionRules.empty();
}

//const std::list<VariableSelectionRule>& ClassSelectionRule::GetFieldSelectionRules()
const std::list<VariableSelectionRule>& ClassSelectionRule::GetFieldSelectionRules() const
{
  return fFieldSelectionRules;
}

void ClassSelectionRule::AddMethodSelectionRule(const FunctionSelectionRule& method)
{
  fMethodSelectionRules.emplace_back(method);
}

bool ClassSelectionRule::HasMethodSelectionRules() const
{
  return !fMethodSelectionRules.empty();
}

void ClassSelectionRule::Print(std::ostream &out) const
{
   out<<"\t\tSelected (line "<< GetLineNumber() <<"): ";
   switch(GetSelected()){
      case BaseSelectionRule::kYes: out<<"Yes"<<std::endl;
         break;
      case BaseSelectionRule::kNo: out<<"No"<<std::endl;
         break;
      case BaseSelectionRule::kDontCare: out<<"Don't Care"<<std::endl;
         break;
      default: out<<"Unspecified"<<std::endl;
   }
   out<<"\t\tAttributes: "<<std::endl;
   PrintAttributes(out,2);

   if (HasFieldSelectionRules()) {
      //out<<"\t\tHas field entries"<<std::endl;
      std::list<VariableSelectionRule> fields = GetFieldSelectionRules();
      std::list<VariableSelectionRule>::iterator fit = fields.begin();
      int j = 0;

      for (; fit != fields.end(); ++fit, ++j)
         {
            out<<"\t\tField "<<j<<":"<<std::endl;
            out<<*fit;
          }
   }
   else {
      out<<"\t\tNo field sel rules"<<std::endl;
   }
   if (HasMethodSelectionRules()) {
      //out<<"\t\tHas method entries"<<std::endl;
      std::list<FunctionSelectionRule> methods = GetMethodSelectionRules();
      std::list<FunctionSelectionRule>::iterator mit = methods.begin();
      int k = 0;

      for (; mit != methods.end(); ++mit, ++k)
         {
            out<<"\t\tMethod "<<k<<":"<<std::endl;
            out<<*mit;
         }
   }
   else {
      out<<"\t\tNo method sel rules"<<std::endl;
   }
}

//const std::list<FunctionSelectionRule>& ClassSelectionRule::GetMethodSelectionRules()
const std::list<FunctionSelectionRule>& ClassSelectionRule::GetMethodSelectionRules() const
{
  return fMethodSelectionRules;
}

bool ClassSelectionRule::IsInheritable() const
{
  return fIsInheritable;
}

void ClassSelectionRule::SetInheritable(bool inherit)
{
  fIsInheritable = inherit;
}

bool ClassSelectionRule::RequestStreamerInfo() const
{
   return fRequestStreamerInfo;
}

void ClassSelectionRule::SetRequestStreamerInfo(bool pl)
{
   fRequestStreamerInfo = pl;
}

bool ClassSelectionRule::RequestNoStreamer() const
{
   return fRequestNoStreamer;
}

void ClassSelectionRule::SetRequestNoStreamer(bool mn)
{
   fRequestNoStreamer = mn;
}

bool ClassSelectionRule::RequestNoInputOperator() const
{
   return fRequestNoInputOperator;
}

void ClassSelectionRule::SetRequestNoInputOperator(bool excl)
{
   fRequestNoInputOperator = excl;
}

void ClassSelectionRule::SetRequestOnlyTClass(bool value)
{
   fRequestOnlyTClass = value;
}

void ClassSelectionRule::SetRequestProtected(bool value)
{
   fRequestProtected = value;
}

void ClassSelectionRule::SetRequestPrivate(bool value)
{
   fRequestPrivate = value;
}

void ClassSelectionRule::SetRequestedVersionNumber(int version)
{
   fRequestedVersionNumber = version;
}

bool ClassSelectionRule::RequestOnlyTClass() const
{
   return fRequestOnlyTClass;
}

bool ClassSelectionRule::RequestProtected() const
{
   return fRequestProtected;
}

bool ClassSelectionRule::RequestPrivate() const
{
   return fRequestPrivate;
}

int ClassSelectionRule::RequestedVersionNumber() const
{
   return fRequestedVersionNumber;
}
