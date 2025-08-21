#ifndef TWO_FLOATS_SNAPSHOT_H
#define TWO_FLOATS_SNAPSHOT_H
#include "Rtypes.h"

class TwoFloats_Snapshot {
public:
   virtual ~TwoFloats_Snapshot() {} // to make dictionary generation happy
   float a = 0;
   float b = 0;
   TwoFloats_Snapshot() {};
   TwoFloats_Snapshot(float d) : a(d), b(2 * d) {};
   void Set(float d)
   {
      a = d;
      b = 2 * d;
   }
   ClassDef(TwoFloats_Snapshot, 1)
};

#endif