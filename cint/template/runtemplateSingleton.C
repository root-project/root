#include "Singleton.h+"

void runtemplateSingleton(bool output=false)
{
   Singleton<int>::Instance().DoIt(output);
   Singleton<double>::Instance().DoIt(output);
}
