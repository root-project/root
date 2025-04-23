#include "vectorHolder.h"
#include "test.C"

void vtest(bool readother=false) {
   const char* testname = "vector";
   typedef vectorHolder holder;

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname,0,readother);
}

void vreadtest(const char*dirname, int nEntry = 0) {
   const char* testname = "vector";
   typedef vectorHolder holder;

   read<holder>(dirname, testname, nEntry, true);
}
