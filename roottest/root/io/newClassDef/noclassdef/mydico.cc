/********************************************************
* mydico.cc
********************************************************/
#include "mydico.h"

#ifdef G__MEMTEST
#undef malloc
#undef free
#endif

extern "C" void G__cpp_reset_tagtablemydico();

extern "C" void G__set_cpp_environmentmydico() {
  G__add_compiledheader("TROOT.h");
  G__add_compiledheader("TMemberInspector.h");
