#include "multisetHolder.h"
#include "test.C"

void ttest() {
   checkHolder<multisetHolder>("multiset");
   write<multisetHolder>("multiset");
   read<multisetHolder>("multiset");
}
