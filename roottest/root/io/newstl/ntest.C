#include "multimapHolder.h"
#include "test.C"

void ntest(bool readother=false) {
   const char* testname = "multimap";
   typedef multimapHolder holder;

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname,0,readother);
}
