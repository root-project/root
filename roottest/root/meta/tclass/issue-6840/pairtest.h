#pragma once

#include "TClass.h"
#include "TError.h"
#include "TFile.h"
#include "TInterpreter.h"
#include "TStreamerElement.h"
#include "TStreamerInfo.h"
#include "TSystem.h"

#include <vector>
#include <list>

namespace edm_test { template <typename T> class FwdPtr; }
class CaloTowerTest;
namespace edm_test { template <> class FwdPtr<CaloTowerTest>; }

struct SameAsShort {
   short fValue;
   operator short() const { return fValue; }
};

struct Contains {
   Contains() {
      fShort.first = '0';
      fSameAs.first = '0';
   }
   std::pair<unsigned char, short>       fShort;
   std::pair<unsigned char, SameAsShort> fSameAs;
};
