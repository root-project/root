/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file strstrm.h
 ************************************************************************
 * Description:
 *  Stub file for making iostream library
 ************************************************************************
 * Copyright(c) 1998       Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__STRSTREAM_H
#define G__STRSTREAM_H

#ifndef __CINT__

#include <strstrea.h>
using namespace std;

#else // __CINT__

#include "iostrm.h"

class strstreambuf : public basic_streambuf<char, char_traits<char> > {
   public:
    typedef char                         char_type;
    typedef char_traits<char>            traits;
    typedef traits::int_type             int_type;
#ifdef __CINT__
    typedef streampos                    pos_type;
    typedef streamoff                    off_type;
#else
    typedef traits::pos_type             pos_type;
    typedef traits::off_type             off_type;
#endif
    strstreambuf(streamsize alsize = 0);
    strstreambuf(void *(*palloc)(size_t), void (*pfree)(void *));
    strstreambuf(char *gnext, streamsize n, char *pbeg = 0);
    strstreambuf(unsigned char *gnext, streamsize n,unsigned char *pbeg = 0);
    strstreambuf(signed char *gnext, streamsize n,signed char *pbeg = 0);
    strstreambuf(const char *gnext, streamsize n);
    strstreambuf(const unsigned char *gnext, streamsize n);
    strstreambuf(const signed char *gnext, streamsize n);
    virtual ~strstreambuf();
    void freeze(bool f = 1);
    char *str();
    int pcount() const;
};

class istrstream : public basic_istream<char, char_traits<char> > {
  public:
    typedef char                         char_type;
    typedef char_traits<char>            traits;
    typedef traits::int_type             int_type;
#ifdef __CINT__
    typedef streampos                    pos_type;
    typedef streamoff                    off_type;
#else
    typedef traits::pos_type             pos_type;
    typedef traits::off_type             off_type;
#endif
    istrstream(const char *s);
    istrstream(const char *s, streamsize n);
    istrstream(char *s);
    istrstream(char *s, streamsize n);
    virtual ~istrstream();
    strstreambuf *rdbuf() const;
    char *str();
};

class ostrstream : public basic_ostream<char, char_traits<char> > {
  public:
    typedef char                               char_type;
    typedef char_traits<char>                  traits;
    typedef traits::int_type                   int_type;
#ifdef __CINT__
    typedef streampos                    pos_type;
    typedef streamoff                    off_type;
#else
    typedef traits::pos_type                   pos_type;
    typedef traits::off_type                   off_type;
#endif
    ostrstream();
    ostrstream(char *s, int n,ios_base::openmode = ios_base::out);
    virtual ~ostrstream();
    strstreambuf *rdbuf() const;
    void freeze(int freezefl = 1);
    char *str();
    int pcount() const;
};

#ifndef __CINT__
class strstream : public basic_iostream<char, char_traits<char> > {
  public:
    typedef char                         char_type;
    typedef char_traits<char>            traits;
    typedef traits::int_type             int_type;
    typedef traits::pos_type             pos_type;
    typedef traits::off_type             off_type;
    strstream();
    strstream(char *s,int n,ios_base::openmode=ios_base::out|ios_base::in);
    void freeze(int freezefl = 1);
    int pcount() const;
    virtual ~strstream();
    strstreambuf *rdbuf() const;
    char *str();
  };
#endif // __CINT__

#endif // __CINT__

#endif // G__STRSTREAM_H
