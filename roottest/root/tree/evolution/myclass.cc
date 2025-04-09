// myclass.cc 

#if REQUIRED_VERSION==1
#include "myclass1.h"
#elif REQUIRED_VERSION==2
#include "myclass2.h"
#endif

ClassImp(myclass) 
