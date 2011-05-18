/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

//-- File booltest.cxx

#ifndef __CINT__
#include <stdio.h>
#endif

#ifdef MISMATCH_REF
void changeByBadRef(int& toChange) {
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
    printf("Changing by ref: result = %d %s\n",result,(result?"True":"False"));

    result = false; //-- to reset value.
    changeByPtr( &result );
    printf("Changing by ptr: result = %s\n",(result?"True":"False"));

#ifdef MISMATCH_REF
    result = false; //-- to reset value.
    changeByBadRef( result );
    printf("Trying bad ref: result = %d %s\n",result,(result?"True":"False"));
#endif
  }

  return 0;
}

//-- End of file

