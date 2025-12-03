#ifndef ClingWorkAroundMissingDynamicScope
{
#else
#include "TTestClass.h"
void runTTestClass()
{
#endif

#ifndef ClingWorkAroundMissingDynamicScope
#ifndef ClingWorkAroundMissingSmartInclude
   #include "TTestClass.h+"
#endif
#endif

   TTestClass obj;
   obj.GetI();
   obj.GetII(0);
   obj.GetIII(0);
   obj.GetPI();
   obj.GetPII(0);
   obj.GetPIII(0);
#ifndef ClingWorkAroundMissingDynamicScope
   return 0;
#endif
}
