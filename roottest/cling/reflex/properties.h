#include "Rtypes.h"

class WithProperties {
public:
   int memWithComment; // a comment
   int memNoIO; //! no I/O
   WithProperties* memWithoutSplitting; //|| no splitting
   int* memPtrAlwaysValid; //-> this pointer always points
   Double32_t memDouble32Range; //[-1,0,12] 12 bits from -1 to 0
};
