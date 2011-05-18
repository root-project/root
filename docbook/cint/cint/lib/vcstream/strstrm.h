/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file strstrm.h
 ************************************************************************
 * Description:
 *  Stub file for making iostream library
 ************************************************************************
 * Copyright(c) 1996       Osamu Kotanigawa
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__STRSTREAM_H
#define G__STRSTREAM_H

#ifndef __CINT__

#ifndef _WINDOWS
#include <strstream.h>
#else
#include <strstrea.h>
#endif

#else

#include "iostrm.h"

class strstreambuf : public streambuf
{
 public: 
  strstreambuf() ;
  strstreambuf(int) ;
  // strstreambuf(void* (*a)(long), void (*f)(void*)) ;
  strstreambuf(char* b, int size, char* pstart = 0 ) ;
  // strstreambuf(unsigned char* b, int size, unsigned char* pstart = 0 ) ;
  // int		pcount();
  void		freeze(int n=1) ;
  char*		str() ;
  ~strstreambuf() ;

 protected:
  virtual int	doallocate() ;
  virtual int	overflow(int) ;
  virtual int	underflow() ;
 public:
  virtual streambuf* setbuf(char*  p, int l) ;
  virtual streampos seekoff(streamoff,ios::seek_dir,int) ;

 public:
  // int isfrozen();
} ;

class istrstream : public istream {
public:
  istrstream(char* str);
  istrstream(char* str, int size ) ;
  istrstream(const char* str);
  istrstream(const char* str, int size);
  ~istrstream() ;
  inline strstreambuf* rdbuf() const ;
  inline char * str() ;
} ;

class ostrstream : public ostream {
public:
  ostrstream(char* str, int size, int=ios::out) ;
  ostrstream() ;
  ~ostrstream() ;
  inline int pcount() const ;
  inline strstreambuf* rdbuf() const ;
  inline char * str() ;
} ;


class strstream : public iostream {
public:
  strstream() ;
  strstream(char* str, int size, int mode) ;
  ~strstream() ;
  inline int pcount() const ;
  inline strstreambuf* rdbuf() const ;
  inline char * str() ;
} ;

#endif
#endif
