// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RCutFlowReport.hxx"
#include "TString.h" // Printf

#include <algorithm>
#include <stdexcept>

namespace ROOT {

namespace RDF {

void RCutFlowReport::Print()
{
   const auto allEntries = fCutInfos.empty() ? 0ULL : fCutInfos.begin()->GetAll();
   for (auto &&ci : fCutInfos) {
      const auto &name = ci.GetName();
      const auto pass = ci.GetPass();
      const auto all = ci.GetAll();
      const auto eff = ci.GetEff();
      const auto cumulativeEff = 100.f * float(pass) / float(allEntries);
      Printf("%-10s: pass=%-10lld all=%-10lld -- eff=%3.2f %% cumulative eff=%3.2f %%", name.c_str(), pass, all, eff, cumulativeEff);
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
