#include "TObject.h"

class Content {
public:
   int fValues;
};
#ifdef __MAKECINT__
#pragma link C++ class Content+;
#endif

class Simple {
public:
   virtual ~Simple() {};
#if VERSION==1
   int fData;
   std::vector<Content> fWillBeMissing;
   ClassDef(Simple,1);
#elif VERSION==2
   float fData;
   ClassDef(Simple,2);
#else
#error VERSION is not set
#endif
};
