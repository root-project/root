#include "Holder.h"
#include "test.C"

void vtest() {
   checkHolder<vectorHolder>();
   write<vectorHolder>("vector.root");
   read<vectorHolder>("vector.root");
}
