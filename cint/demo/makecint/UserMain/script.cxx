/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

void script(const char* msg="default argument") {
  printf("%s\n",msg);
  f1(1234);
  f2(3.141592);
}

int main() {
  script("Calling from interpreted main function");
}
