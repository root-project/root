#ifndef __CINT__
#include <stdio.h>
#endif

int main(int argc,char** argv) {

   int i = 0;
   printf("Arg %d\n",argc,argv);
   for(i=0;i<argc;i++) {
      const char *argument = argv[i];
      printf("Arg #%d \n",i,argument);
      printf("Arg #%d %s\n",i,argument);
   }
   return 0;
}
