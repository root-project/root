#include "dequeHolder.h"
#include "test.C"

void vtest() {
   checkHolder<dequeHolder>();
   write<dequeHolder>("deque.root");
   read<dequeHolder>("deque.root");
}
