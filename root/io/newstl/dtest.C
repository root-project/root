#include "dequeHolder.h"
#include "test.C"

void dtest() {
   checkHolder<dequeHolder>("deque");
   write<dequeHolder>("deque");
   read<dequeHolder>("deque");
}
