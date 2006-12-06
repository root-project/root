/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

//-- File booltest.cxx

//#include <iostream>
//using namespace std;
#include <stdio.h>

#if 0
void changeByRef(int& toChange) {
  toChange = 123;
}
#endif

void changeByRef(bool& toChange)
{
  //printf("i1 result = %s\n",(toChange?"True":"False"));
  toChange = true;
  //printf("i2 result = %s\n",(toChange?"True":"False"));
}

void changeByPtr(bool* toChange)
{
  *toChange = true;
}

int main()
{
  bool result;
  for(int i=0;i<3;i++) {
    result = false; //-- to reset value.
    changeByRef( result );
    printf("result = %s\n",(result?"True":"False"));

    result = false; //-- to reset value.
    changeByPtr( &result );
    printf("result = %s\n",(result?"True":"False"));
  }

  return 0;
}

//-- End of file

