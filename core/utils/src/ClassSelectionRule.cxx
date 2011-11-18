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

void ClassSelectionRule::AddFieldSelectionRule(VariableSelectionRule field)
{
  fFieldSelectionRules.push_back(field);
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

void ClassSelectionRule::AddMethodSelectionRule(FunctionSelectionRule method)
{
  fMethodSelectionRules.push_back(method);
}

bool ClassSelectionRule::HasMethodSelectionRules() const
{
  return !fMethodSelectionRules.empty();
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

bool ClassSelectionRule::HasPlus() const
{
   return fPlus;
}

void ClassSelectionRule::SetPlus(bool pl)
{
   fPlus = pl;
}

bool ClassSelectionRule::HasMinus() const
{
   return fMinus;
}

void ClassSelectionRule::SetMinus(bool mn)
{
   fMinus = mn;
}

bool ClassSelectionRule::HasExclamation() const
{
   return fExclamation;
}

void ClassSelectionRule::SetExclamation(bool excl)
{
   fExclamation = excl;
}

void ClassSelectionRule::SetRequestOnlyTClass(bool value)
{
   fRequestOnlyTClass = value;
}

bool ClassSelectionRule::RequestOnlyTClass() const
{
   return fRequestOnlyTClass;
}

bool ClassSelectionRule::RequestNoStreamer() const
{
   return HasMinus();
}

bool ClassSelectionRule::RequestNoInputOperator() const
{
   return HasExclamation();
}

bool ClassSelectionRule::RequestStreamerInfo() const
{
   return HasPlus();
}

