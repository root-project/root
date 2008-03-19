#include <stdio.h>

void switch_test()
{
   for (int i = 0; i < 4; ++i) {
      switch (i) {
         case 0:
         case 1:
            printf("i: %d, case 0 or 1\n", i);
            break;
         case 2:
            printf("i: %d, case 2\n", i);
            break;
         default:
            printf("i: %d, case default\n", i);
      }
   }
   int j = 1;
   switch(j){
   case (0):
      printf("Outer Case (0)\n");
      break;
   case(1):
      printf("Outer Case (1)\n");
      break;
   case(2):
      printf("Outer Case (2)\n");
      break;
   case(3):
      printf("Outer Case (3)\n");
      break;
   default:
      printf("Outer Case with parenthesis failed to find %d\n",j);
   }
   for (int type = 0; type < 4; ++type) {
      switch(type){
         case (0):
            printf("Case (0)\n");
            break;
         case(1):
            printf("Case (1)\n");
            break;
         case(2):
            printf("Case (2)\n");
            break;
         case(3):
            printf("Case (3)\n");
            break;
         default:
            printf("Case with parenthesis failed to find %d\n",type);
      }
   }
   
}

int main()
{
   switch_test();
   return 0;
}

