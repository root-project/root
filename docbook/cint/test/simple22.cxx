#include "simple22.h"

int myclass::myfunc() {
   printf("executing myfunc\n");
}

int main() 
{
   myclass a;
   a.myfunc();
   return 0;
}

