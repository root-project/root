#include "setHolder.h"
#include "test.C"

void stest() {
   checkHolder<setHolder>("set");
   write<setHolder>("set");
   read<setHolder>("set");
}
