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

