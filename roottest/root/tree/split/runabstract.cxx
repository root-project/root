#include "abstract.cxx+"

int runabstract(int mode = 1) {
   // 0: write the file
   // 1: read the file

   switch(mode) {
   case 0: writefile();
      return 0;
      break;
   case 1: readfile();
      return 0;
      break;
   }
   return 1;
}
