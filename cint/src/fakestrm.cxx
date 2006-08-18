/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file fakestrm.cxx
 ************************************************************************
 * Description:
 *  Dummy function for fake iostream library
 ************************************************************************
 * Author                  Masaharu Goto 
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

extern "C" void G__cpp_setupG__stream() {
}

extern "C" void G__redirectcout(const char* filename) {
}
extern "C" void G__unredirectcout() {
}
extern "C" void G__redirectcerr(const char* filename) {
}
extern "C" void G__unredirectcerr() {
}
extern "C" void G__redirectcin(const char* filename) {
}
extern "C" void G__unredirectcin() {
}
