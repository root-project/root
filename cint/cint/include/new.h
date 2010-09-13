/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/****************************************************************
* new.h
*****************************************************************/
#ifndef G__NEW_H
#define G__NEW_H

#include <stdio.h>

// work around ::operator new(size) 
void* operator new(size_t s) {
  return new char[s];
}
//void* operatornew(unsigned long s) {return malloc(s);}

// work around ::operator delete(buffer) 
void operator delete(void* p) {
  if(p) delete p; 
}

// set_new_handler is a dummy function that is provided to
// allow interpretation of the Standard Template Library (STL).
typedef void (*new_handler)();
new_handler set_new_handler(new_handler x) {return x;}
new_handler set_new_handler(long x) {return((new_handler)x);}

#endif
