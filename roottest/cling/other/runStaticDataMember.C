#include "runStaticDataMember.h"

#include <stdio.h>

using namespace std;

A::A()
: x(0)
{
   printf("A::A()\n");
}

A::~A()
{
   printf("A::~A()\n");
}

A B::a;

B::B()
{
   printf("B::B()\n");
}

B::~B()
{
   printf("B::~B()\n");
}

void runStaticDataMember()
{
   B b;
}

#ifndef __MAKECINT__
int main()
{
   runStaticDataMember();
   return 0;
}
#endif // __MAKECINT__
