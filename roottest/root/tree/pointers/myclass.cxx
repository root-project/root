#include "myclass.h"

void verify() {
   myclass *m = new myclass();
   m->verify();
   m->set();
   m->verify();
   delete m;
}
