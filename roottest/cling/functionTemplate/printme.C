#include <iostream>
//#include "t.h"

void printme(const t& o) {
   std::cout << flush;
   double d=o.get<double>();
   int i=o.get<int>();
   int j=o.get<float>();

   std::cout << "t now " << d << " " << i << " " << j << std::endl;
   std::cout << flush;
   float v = o.getfloat();
   std::cout << "t now " << v << std::endl;
   std::cout << flush;
}
