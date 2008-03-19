#ifndef __CINT__
#include <stdio.h>
#endif
//#include <iostream>

class myclass {
public:
   myclass(int a) : i(a) {};
   int i;
};

void printing(myclass &obj,int j)
{
   printf("printing %x %d %d\n", 0x0 /* &obj */,obj.i,j);
}

myclass& operator<<(myclass &obj, int j)
{
   printf("operating %x %d %d\n", 0x0 /* &obj */,obj.i,j);
   return obj;
}

int main()
{
   int i = 33;
   myclass a(22);
   printing(a,i);
   a << 44;
   //cout << i;
   //cout << endl;
   return 0;
}
