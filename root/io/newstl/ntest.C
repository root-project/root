#include "multimapHolder.h"
#include "test.C"

void ntest() {
   checkHolder<multimapHolder>("multimap");
   write<multimapHolder>("multimap");
   read<multimapHolder>("multimap");
}
