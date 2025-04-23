#ifndef HLTReco_TriggerTypeDefs_h
#define HLTReco_TriggerTypeDefs_h

/** \class trigger::TriggerTypeDefs
 *
 *  Misc. common simple typedefs
 *
 *  $Date: 2009/11/13 17:22:36 $
 *  $Revision: 1.10 $
 *
 *  \author Martin Grunewald
 *
 */

#include <vector>
#include <stdint.h>

namespace trigger
{

  typedef uint16_t size_type;
  typedef std::vector<size_type> Keys;

  typedef std::vector<int>       Vids;

  enum TriggerObjectType  {

    /// enum start value shifted to 81 so as to avoid clashes with PDG codes

    /// L1 - using cases as defined in enum L1GtObject, file:
    /// DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

    TriggerL1Mu           = -81,
    TriggerL1NoIsoEG      = -82,
    TriggerL1IsoEG        = -83,
    TriggerL1CenJet       = -84,
    TriggerL1ForJet       = -85,
    TriggerL1TauJet       = -86,
    TriggerL1ETM          = -87,
    TriggerL1ETT          = -88,
    TriggerL1HTT          = -89,
    TriggerL1HTM          = -90,
    TriggerL1JetCounts    = -91,
    TriggerL1HfBitCounts  = -92,
    TriggerL1HfRingEtSums = -93,
    TriggerL1TechTrig     = -94,
    TriggerL1Castor       = -95,
    TriggerL1BPTX         = -96,
    TriggerL1GtExternal   = -97,

    /// HLT

    TriggerPhoton         = +81,
    TriggerElectron       = +82,
    TriggerMuon           = +83,
    TriggerTau            = +84,
    TriggerJet            = +85,
    TriggerBJet           = +86,
    TriggerMET            = +87,
    TriggerTET            = +88,
    TriggerTHT            = +89,
    TriggerMHT            = +90,
    TriggerTrack          = +91,
    TriggerCluster        = +92,
    TriggerMETSig         = +93,
    TriggerELongit        = +94,
    TriggerMHTSig         = +95,
    TriggerHLongit        = +96

  };

}

class dummy{};

#endif

