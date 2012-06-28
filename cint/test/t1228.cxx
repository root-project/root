/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <ertti.h>

typedef bool Bool_t;
const Bool_t kTrue = true;
const Bool_t kFalse = false;

void f(int a,Bool_t b) {
  printf("%d %d\n",a,b);
}

void Exe(const char* fname,const char* args) {
  G__ClassInfo g;
  long dmy=0;
  G__CallFunc cf;
  cf.SetFunc(&g,fname,args,&dmy);
  printf("IsValid=%d  ",cf.IsValid());
  cf.Exec(&dmy);
  cf.ResetArg();
}

int main() {
  Exe("f","10,true");
  Exe("f","20,1");
  Exe("f","30,kTrue");
  Exe("f","40,false");
  Exe("f","50,0");
  Exe("f","60,kFalse");
}


