// @(#)root/core/utils:$Id: VariableSelectionRule.h 28529 2009-05-11 16:43:35Z pcanal $
// Author: Velislava Spasova September 2010

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef R__VARIABLESELECTIONRULE_H
#define R__VARIABLESELECTIONRULE_H

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// VariableSelectionRule                                                //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "BaseSelectionRule.h"
#include <string>

class VariableSelectionRule final : public BaseSelectionRule
{
public:
   VariableSelectionRule(ESelect sel) : BaseSelectionRule(sel) {}
   VariableSelectionRule(long index, cling::Interpreter &interp, const char* selFileName = "", long lineno=-1) : BaseSelectionRule(index, interp, selFileName, lineno) {}
   VariableSelectionRule(long index, ESelect sel, std::string attributeName, std::string attributeValue, cling::Interpreter &interp, const char* selFileName = "", long lineno=1)
       : BaseSelectionRule(index, sel, attributeName, attributeValue, interp,selFileName, lineno){}

   void Print(std::ostream &out) const;
};

typedef VariableSelectionRule FunctionSelectionRule; // Function selection rules are the same as Variable selection rules
typedef VariableSelectionRule EnumSelectionRule;     // Enum selection rules are the same as Variable selection rules

#endif
