// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCUTFLOWREPORT
#define ROOT_TCUTFLOWREPORT

#include "RtypesCore.h"
#include "TString.h"

#include <string>
#include <vector>

namespace ROOT {

namespace Detail {
namespace TDF {
class TFilterBase;
} // End NS TDF
} // End NS Detail

namespace Experimental {
namespace TDF {

class TCutInfo {
   friend class TCutFlowReport;
   friend class ROOT::Detail::TDF::TFilterBase;

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

class TCutFlowReport {
   friend class ROOT::Detail::TDF::TFilterBase;

private:
   std::vector<TCutInfo> fCutInfos;
   void AddCut(TCutInfo &&ci) { fCutInfos.emplace_back(std::move(ci)); };

public:
   void Print();
   const TCutInfo &operator[](std::string_view cutName);
   std::vector<TCutInfo>::const_iterator begin() const { return fCutInfos.begin(); }
   std::vector<TCutInfo>::const_iterator end() const { return fCutInfos.end(); }
};

} // End NS TDF
} // End NS Experimental
} // End NS ROOT

#endif
