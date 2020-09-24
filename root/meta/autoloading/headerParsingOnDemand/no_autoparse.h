#pragma once

#include <map>
#include <TFile.h>
#include <TTree.h>


#ifdef ITEM_V10
// If this was also in the reco::Muon namespace, since at reading
// time we will *intentionally* not have any trace (beside the information
// the file's StreamerInfo) of this enum, the search for the enum (reco::Muon::OtherTrackType)"
// will trigger auto-parsing (to try to find it inside the header files).
enum OtherTrackType { kOtherDefault = 1 };
#endif

namespace reco {
using TrackRef = int;

struct Muon {
public:
   enum MuonTrackType { kDefault = 0 };

   typedef std::map<MuonTrackType, reco::TrackRef> MuonTrackRefMap;

   int             id_;
   MuonTrackRefMap refittedTrackMap_;   

#ifdef ITEM_V10
   typedef std::map<OtherTrackType, reco::TrackRef> OtherTrackRefMap;
   OtherTrackRefMap willbeEmulatedMap_;
#endif

};
}

namespace edm {
struct Value {};
}

namespace edmNew {
namespace dstvdetails {
struct DetSetVectorTrans {
struct Item {
#ifdef ITEM_V10
  int fOldValue;
  ClassDefNV(Item, 10);
#else
  float fNewValue;
  ClassDefNV(Item, 11);
#endif
};
};
}
}

void writer(const char *filename = "no_autoparse.root");

