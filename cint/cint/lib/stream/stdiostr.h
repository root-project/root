/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file stdiostream.sut.h
 ************************************************************************
 * Description:
 *  Stub file for making iostream library
 ************************************************************************
 * Copyright(c) 1991~2001,  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__STDSTREAM_H
#define G__STDSTREAM_H

#ifndef __CINT__

#ifndef G__NEWSTDHEADER
#if defined(__APPLE__)
#include <iostream.h>
#elif defined(_WINDOWS)
#include <stdiostr.h>
#else
#include <stdiostream.h>
#endif
#endif

#else

#include "iostrm.h"
#include <stdio.h>

#ifdef G__NEVER
class stdiobuf : public streambuf {
  /*** stdiobuf is obsolete, should be avoided ***/
 public: // Virtuals
  virtual int	overflow(int=EOF);
  virtual int	underflow();
  virtual int	sync() ;
  virtual streampos seekoff(streamoff,ios::seek_dir,int) ;
  virtual int	pbackfail(int c);
 public:
  stdiobuf(FILE* f) ;
  FILE*		stdiofile() { return fp ; }
  virtual		~stdiobuf() ;
};
#endif

#ifdef G__NEVER
class stdiostream : public ios {
public:
  stdiostream(FILE*) ;
  ~stdiostream() ;
  stdiobuf*	rdbuf() ;
 private:
  stdiobuf	buf ;
};
#endif

#endif
#endif

