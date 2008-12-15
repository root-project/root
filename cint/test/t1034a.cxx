#include "stdio.h"
int main() 
{
   for(int i=0;i<2;i++) {
      // sizeof
      printf("sizeof(long long)=%ld\n",sizeof(long long));
      printf("sizeof(long long int)=%ld\n",sizeof(long long int));
      printf("sizeof(unsigned long long)=%ld\n",sizeof(unsigned long long));
      printf("sizeof(unsigned long long int)=%ld\n",sizeof(unsigned long long int));
      printf("sizeof(long double)=%ld\n",sizeof(long double));
   }
   return 0;
}
