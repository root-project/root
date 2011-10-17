// RUN: %cling %s

#include <stdio.h>

const char* defaultArgV[] = {"A default argument", "", 0};

int test_01(int argc=12, const char** argv = defaultArgV)
{
  int i;
  for( i = 0; i < 5; ++i )
    printf( "Hello World #%d\n", i );
  return 0;
}
