#include "t.h"
#include "printme.C"

int main(int, char*[]) {
   t o; printme(o);
   o.set(12); printme(o);
   o.set(42.33); printme(o);
   return 0;
}
