#include "setHolder.h"
#include "test.C"

void stest(bool readother=false) {
   const char* testname = "set";
   typedef setHolder holder;

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname,0,readother);
}
