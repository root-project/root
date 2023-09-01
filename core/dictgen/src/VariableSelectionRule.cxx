// @(#)root/core/utils:$Id: VariableSelectionRule.cxx 41697 2011-11-01 21:03:41Z pcanal $
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
// VariableSelectionRule                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "VariableSelectionRule.h"
#include <iostream>

void VariableSelectionRule::Print(std::ostream &out) const
{
   out<<"\t\tSelected: ";
   switch(GetSelected()){
   case BaseSelectionRule::kYes: out<<"Yes"<<std::endl;
      break;
   case BaseSelectionRule::kNo: out<<"No"<<std::endl;
      break;
   case BaseSelectionRule::kDontCare: out<<"Don't Care"<<std::endl;
      break;
   default: out<<"Unspecified"<<std::endl;
   }
   PrintAttributes(out,3);
}
