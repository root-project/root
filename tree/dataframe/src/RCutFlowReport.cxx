// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCutFlowReport.hxx"

#include <algorithm>
#include <stdexcept>

namespace ROOT {

namespace RDF {

void RCutFlowReport::Print()
{
   for (auto &&ci : fCutInfos) {
      auto &name = ci.GetName();
      auto pass = ci.GetPass();
      auto all = ci.GetAll();
      auto eff = ci.GetEff();
      Printf("%-10s: pass=%-10lld all=%-10lld -- %8.3f %%", name.c_str(), pass, all, eff);
   }
}
const TCutInfo &RCutFlowReport::operator[](std::string_view cutName)
{
   if (cutName.empty()) {
      throw std::runtime_error("Cannot look for an unnamed cut.");
   }
   auto pred = [&cutName](const TCutInfo &ci) { return ci.GetName() == cutName; };
   const auto ciItEnd = fCutInfos.end();
   const auto it = std::find_if(fCutInfos.begin(), ciItEnd, pred);
   if (ciItEnd == it) {
      std::string err = "Cannot find a cut called \"";
      err += cutName;
      err += "\". Available named cuts are: \n";
      for (auto &&ci : fCutInfos) {
         err += " - " + ci.GetName() + "\n";
      }
      throw std::runtime_error(err);
   }
   return *it;
}

} // End NS RDF

} // End NS ROOT
