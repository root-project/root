#include "f1.C"

void f() {
#if 0
  fprintf(stderr,"%s \n",MyTemplate<int>::Class_Name());
#else
  MyTemplate<int> a;
  MyTemplate<double> b;
  fprintf(stderr,"%s \n",a.Class_Name());
  fprintf(stderr,"%s \n",b.Class_Name());
   MyTemplate<double> * bb = a.getThis();
#endif
  
}


