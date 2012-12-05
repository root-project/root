#include <iostream>

void assertManipulators() {
   double mark = 5.0;
#ifndef ClingWorkAroundJITandInline
   std::cout << " mark is " << fixed << mark << std::endl;
   std::cout << " mark is " << scientific << mark << std::endl;
#else
   std::cout << " mark is " << mark << std::endl;
   std::cout << " mark is " << mark << std::endl;
#endif
}
