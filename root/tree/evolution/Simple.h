#include "TObject.h"
class Simple {
public:
#if VERSION==1
   int fData;
   ClassDef(Simple,1);
#elif VERSION==2
   float fData;
   ClassDef(Simple,2);
#else
#error VERSION is not set
#endif
};
