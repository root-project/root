#ifndef ORDERCLASS_H

#ifdef VERSION1
#include "MyClass_v1.h"
#elif VERSION2
#include "MyClass_v2.h"
#endif

#endif

#include <iostream>

void MyClass::Print(Option_t* /* option */) const
{
   std::cout << "ver: " << ver << "\n";
   std::cout << "transientMember: " << transientMember << "\n";
   //std::cout << "arr[0]: " << fArray[0] << "\n";
   //std::cout << "arr[1]: " << fArray[1] << "\n";
}
