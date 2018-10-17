#ifndef __RErrorIgnoreRAII__
#define __RErrorIgnoreRAII__

#include "TError.h"

class RErrorIgnoreRAII{
   int fPrevIgnoreLevel;
public:
   RErrorIgnoreRAII(const int level = kFatal) : fPrevIgnoreLevel(gErrorIgnoreLevel)
   {
      gErrorIgnoreLevel = level;
   }
   ~RErrorIgnoreRAII()
   {
      gErrorIgnoreLevel = fPrevIgnoreLevel;
   }
};

#endif
