#ifndef ATHINDEX_HH
#define ATHINDEX_HH

#include "TObject.h"
#include <string>
using namespace std;

class AthIndex : public TObject {
public:
  enum indexType { noInit         = 0,
         indexWire      = 1,
         indexLayer     = 2,
         indexMezzanine = 3,
         indexStation   = 4 };

  indexType type;
  int stationName;
  int stationEta;
  int stationPhi;
  int multiLayer;
  int tubeLayer;
  int tube;
  int mezzanine;

  AthIndex() {};
  virtual ~AthIndex() {};

  ClassDef(AthIndex,1) // Holder class for the Athena indices
};

#endif
