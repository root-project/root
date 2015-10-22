#ifndef RecoTracker_TkSeedGenerator_ClusterChecker_H
#define RecoTracker_TkSeedGenerator_ClusterChecker_H

#include "DetSetVectorNew.h"
#include "SiStripCluster.h"

namespace edm { class Event; class ParameterSet; }

namespace reco { namespace utils {
   struct ClusterTotals {
      ClusterTotals() : strip(0), pixel(0), stripdets(0), pixeldets(0) {}
      int strip; /// number of strip clusters
      int pixel; /// number of pixel clusters
      int stripdets; /// number of strip detectors with at least one cluster
      int pixeldets; /// number of pixel detectors with at least one cluster
   };
} }

namespace edm {
   template <typename T> class EDGetTokenT {};
}

class ClusterChecker {
public:
   edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > token_sc;
   // edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > token_pc;
};

#endif
