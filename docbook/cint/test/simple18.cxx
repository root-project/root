#ifndef __CINT__
#include <stdio.h>
#endif

class myclass 
{
public:
   static void myfunc(int i);
   static int myvar;
};

void myclass::myfunc(int i)
{
   printf("running myfunc %d\n",i);
}

int myclass::myvar = 0;

int main() 
{
   myclass::myfunc(3);
   return 0;
}
