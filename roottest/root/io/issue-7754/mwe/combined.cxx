
#include "src/SpectrometerChannelID.cc"
#include "src/TDetectorVEvent.cc"
#include "src/TDetectorVHit.cc"
#include "src/TSpectrometerEvent.cc"
#include "src/TSpectrometerHit.cc"
#include "src/TVDigi.cc"
#include "src/TVEvent.cc"
#include "src/TVHit.cc"

#include "PersistencyLinkDef.hh"

#include "read_updated.cxx"

int combined(int vers) {
   read_updated(vers);
   return 0;
}