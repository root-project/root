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
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

extern "C" void G__cpp_setupG__stream() {
}

#ifndef G__OLDIMPLEMENTATION1635
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
#endif /* 1635 */
