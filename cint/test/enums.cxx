#if defined(interp) && defined(makecint)
#include "test.dll"
#else
#include "enums.h"
#endif

#ifndef __CINT__
#include <stdio.h>
#endif

int main()
{
  printf("value is %d\n",kOne);
  return 0;
}
