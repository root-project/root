#include "vectorHolder.h"
#include "test.C"

void vtest() {
   const char* testname = "vector";
   checkHolder<vectorHolder>(testname);
   write<vectorHolder>(testname);
   read<vectorHolder>(testname);
}
