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

bool ClassSelectionRule::HasFieldSelectionRules()
{
  return !fFieldSelectionRules.empty();
}

//const std::list<VariableSelectionRule>& ClassSelectionRule::GetFieldSelectionRules()
std::list<VariableSelectionRule>& ClassSelectionRule::GetFieldSelectionRules()
{
  return fFieldSelectionRules;
}

void ClassSelectionRule::AddMethodSelectionRule(FunctionSelectionRule method)
{
  fMethodSelectionRules.push_back(method);
}

bool ClassSelectionRule::HasMethodSelectionRules()
{
  return !fMethodSelectionRules.empty();
}

//const std::list<FunctionSelectionRule>& ClassSelectionRule::GetMethodSelectionRules()
std::list<FunctionSelectionRule>& ClassSelectionRule::GetMethodSelectionRules()
{
  return fMethodSelectionRules;
}

bool ClassSelectionRule::IsInheritable()
{
  return fIsInheritable;
}

void ClassSelectionRule::SetInheritable(bool inherit)
{
  fIsInheritable = inherit;
}

bool ClassSelectionRule::HasPlus()
{
   return fPlus;
}

void ClassSelectionRule::SetPlus(bool pl)
{
   fPlus = pl;
}

bool ClassSelectionRule::HasMinus()
{
   return fMinus;
}

void ClassSelectionRule::SetMinus(bool mn)
{
   fMinus = mn;
}

bool ClassSelectionRule::HasExclamation()
{
   return fExclamation;
}

void ClassSelectionRule::SetExclamation(bool excl)
{
   fExclamation = excl;
}


