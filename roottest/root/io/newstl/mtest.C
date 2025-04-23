#include "mapHolder.h"
#include "test.C"

void mtest(bool readother=false) {
   const char* testname = "map";
   typedef mapHolder holder;

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname,0,readother);
}

void mtest_readonly(bool readother=false) 
{
   const char* testname = "map";
   typedef mapHolder holder;

   std::cout << "Running read test " << testname << std::endl;
   read<holder>(testname,0,readother);
}
