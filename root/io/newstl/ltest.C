#include "listHolder.h"
#include "test.C"

void ltest() {
   checkHolder<listHolder>("list");
   write<listHolder>("list");
   read<listHolder>("list");
}
