/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef const char Option_t;
class TString {
  char dum[4];
  char buf[100];
 public:
  TString(const char* s="") { 
    strcpy(dum,"XXX");
    strcpy(buf,s); 
  }
  TString(const TString& x) { 
    strcpy(dum,"YYY");
    strcpy(buf,x.buf); 
  }
  operator const char*() {
    printf("TString::operator const char*()\n");
    return(buf);
  }
};

class TTree {
 public:
  void Draw(Option_t* x) {
    printf("TTree::Draw(%s)\n",x);
  }
  void Draw(const char* a, const char* b) {
    printf("TTree::Draw(%s,%s)\n",a,b);
  }
};


