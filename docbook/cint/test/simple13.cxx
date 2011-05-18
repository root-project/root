#ifndef __CINT__
#include "stdio.h"
#include "string.h"
#endif
int myfunc() {
   return 3;
}
bool mycheck() {
   if (strcmp("test","test")) {
      return true;
   } else {
      return false;
   }
}
int main(int argc,char** argv) {
   int i = 0;
   printf("Arg %d\n",argc,argv);
   for(i=0;i<argc;i++) {
      const char *argument = argv[i];
//      printf("Arg #%d %p\n",i,argument);
      printf("Arg #%d %s\n",i,argument);
   }
   int r = myfunc();
   printf("myfunc()==%d\n",r);
   if (myfunc()) {
      printf("myfunc()==%d\n",r);
   }
   if (mycheck()) {
      printf("mycheck return true\n");
   } 
}
