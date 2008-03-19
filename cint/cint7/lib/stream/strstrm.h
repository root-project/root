/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file stream.sut.h
 ************************************************************************
 * Description:
 *  Stub file for making iostream library
 ************************************************************************
 * Copyright(c) 1991~1999,  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__STRSTREAM_H
#define G__STRSTREAM_H

#ifndef __CINT__

#ifdef G__NEWSTDHEADER
#include <sstream>
#else
#ifndef _WINDOWS
#include <strstream.h>
#else
#include <strstrea.h>
#endif
#endif

#else

#include "iostrm.h"

class strstreambuf : public streambuf
{
 public: 
  strstreambuf() ;
  strstreambuf(int) ;
  // strstreambuf(void* (*a)(long), void (*f)(void*)) ;

  // alpha cxx didn't like below
  //strstreambuf(char* b, int size, char* pstart = 0 ) ;
  strstreambuf(char* b, int size, char* pstart) ;

  // strstreambuf(unsigned char* b, int size, unsigned char* pstart = 0 ) ;
  // int		pcount();
  void		freeze(int n=1) ;
  char*		str() ;
			~strstreambuf() ;

 public: /* virtuals  */
  virtual int	doallocate() ;
 protected:
  virtual int	overflow(int) ;
  virtual int	underflow() ;
 public:
  virtual streambuf* setbuf(char*  p, int l) ;
  virtual streampos seekoff(streamoff,ios::seek_dir,int) ;

 public:
  // int		isfrozen() { return froozen; }
} ;

class strstreambase : public virtual ios {
 public:
  strstreambuf*	rdbuf() ;
 protected:	
  strstreambase(char*, int, char*) ;
  strstreambase() ;
  ~strstreambase() ;
 private:
  strstreambuf	buf ; 
} ;

class istrstream : public strstreambase, public istream {
public:
  istrstream(char* str);
  istrstream(char* str, int size ) ;
  istrstream(const char* str);
  istrstream(const char* str, int size);
  ~istrstream() ;
} ;

class ostrstream : public strstreambase, public ostream {
public:
  ostrstream(char* str, int size, int=ios::out) ;
  ostrstream() ;
  ~ostrstream() ;
  char*		str() ;
  int		pcount() ;
} ;


class strstream : public strstreambase, public iostream {
public:
  strstream() ;
  strstream(char* str, int size, int mode) ;
  ~strstream() ;
  char*		str() ;
} ;

#endif
#endif
