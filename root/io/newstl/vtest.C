#include "Holder.h"
#include "test.C"

void vtest() {
   write<vectorHolder>("vector.root");
   read<vectorHolder>("vector.root");
}
