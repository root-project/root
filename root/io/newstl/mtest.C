#include "mapHolder.h"
#include "test.C"

void mtest() {
   checkHolder<mapHolder>("map");
   write<mapHolder>("map");
   read<mapHolder>("map");
}
