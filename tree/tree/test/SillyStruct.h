#include "Rtypes.h"
#include "TObject.h"

/**
 * The SillyStruct has no purpose except to provide
 * inputs to the test cases.
 */

class SillyStruct : public TObject {

public:
   float  f;
   int    i;
   double d;

   ClassDef(SillyStruct, 1)
};
