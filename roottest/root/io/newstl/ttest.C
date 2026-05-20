#include "multisetHolder.h"
#include "test.C"

void ttest(bool readother=false) {
   const char* testname = "multiset";
   typedef multisetHolder holder;   

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname,0,readother);
}
