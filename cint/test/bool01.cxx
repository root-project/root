#include <cstdio>

using namespace std;

int main()
{
   bool b = 0;
   bool* p = &b;
   bool d;
   d = *p;
   printf("*p: %d\n", *p);
   printf("d: %d\n", d);
   return 0;
}

