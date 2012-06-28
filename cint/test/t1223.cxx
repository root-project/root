/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

// Start of MyOpClass.C
class MyOpClass {
public:
   const char* operator[](const char *s) {
      fprintf(stdout,"calling operator[] with index %s\n",s);
      return s;
   }
   MyOpClass &value() { return *this; }
};
// End of MyOpClass.C


int main() {
  MyOpClass obj;
  obj.value()["33"];
  obj.value()["33"];
  return 0;
}

