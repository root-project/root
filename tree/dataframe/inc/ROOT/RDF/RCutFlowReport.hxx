// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RCUTFLOWREPORT
#define ROOT_RCUTFLOWREPORT

#include "RtypesCore.h"
#include "TString.h"

#include <string>
#include <vector>

namespace ROOT {

namespace Detail {
namespace RDF {
class RFilterBase;
} // End NS RDF
} // End NS Detail

namespace RDF {

class TCutInfo {
   friend class RCutFlowReport;
   friend class ROOT::Detail::RDF::RFilterBase;

private:
   const std::string fName;
   const ULong64_t fPass;
   const ULong64_t fAll;
   TCutInfo(const std::string &name, ULong64_t pass, ULong64_t all) : fName(name), fPass(pass), fAll(all) {}

public:
   const std::string &GetName() const { return fName; }
   ULong64_t GetAll() const { return fAll; }
   ULong64_t GetPass() const { return fPass; }
   float GetEff() const { return 100.f * (fPass / float(fAll)); }
};

class RCutFlowReport {
   friend class ROOT::Detail::RDF::RFilterBase;

private:
   std::vector<TCutInfo> fCutInfos;
   void AddCut(TCutInfo &&ci) { fCutInfos.emplace_back(std::move(ci)); };

public:
   using const_iterator = typename std::vector<TCutInfo>::const_iterator;
   void Print();
   const TCutInfo &operator[](std::string_view cutName);
   const TCutInfo &At(std::string_view cutName) { return operator[](cutName); }
   const_iterator begin() const { return fCutInfos.begin(); }
   const_iterator end() const { return fCutInfos.end(); }
};

} // End NS RDF
} // End NS ROOT

#endif
