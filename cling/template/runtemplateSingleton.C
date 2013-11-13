#ifdef ClingWorkAroundMissingSmartInclude
#include "Singleton.h"
#else
#include "Singleton.h+"
#endif

void runtemplateSingleton(bool output=false)
{
   Singleton<int>::Instance().DoIt(output);
   Singleton<double>::Instance().DoIt(output);
}
