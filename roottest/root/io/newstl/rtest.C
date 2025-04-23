#include "rvecHolder.h"
#include "test.C"

void rtest(bool readother=false) {
   const char* testname = "rvec";
   typedef rvecHolder holder;

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname,0,readother);
}

void rreadtest(const char*dirname, int nEntry = 0) {
   const char* testname = "rvec";
   typedef rvecHolder holder;

   read<holder>(dirname, testname, nEntry, true);
}
