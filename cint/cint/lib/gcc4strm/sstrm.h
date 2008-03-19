/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file sstrm.h
 ************************************************************************
 * Description:
 *  Stub file for making iostream library
 ************************************************************************
 * Copyright(c) 2007 Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__SSTREAM_H
#define G__SSTREAM_H

#ifndef __CINT__

#include <sstream>
using namespace std;

#else // __CINT__

#if !(G__GNUC>=3)
#error This header must be processed with a Cint compiled with gcc3.x
#endif

#include <string>
//#include <memory>
class allocator<char>;
class allocator<wchar_t>;
#include "iostrm.h"

template<class charT, class traits, class Allocator>
class basic_stringbuf : public basic_streambuf<charT, traits>
{
 public:
  typedef charT                                    char_type;
  typedef traits::int_type               int_type;
  typedef traits::pos_type               pos_type;
  typedef traits::off_type               off_type;
  typedef traits                                   traits_type;
  
  typedef basic_ios<charT, traits>                 ios_type;
#ifdef __CINT__
#if 0 && !(G__GNUC>=3)
  typedef string  string_type;
#else
  typedef string  __string_type;
  //typedef basic_string<charT, traits, Allocator >  __string_type;
#endif
#else
  typedef basic_string<charT, traits, Allocator >  string_type;
#endif
  
  explicit basic_stringbuf(ios_base::openmode which = 
			   ios_base::in | ios_base::out );
  
  explicit basic_stringbuf(const __string_type& str,
			   ios_base::openmode which = 
			   ios_base::in | ios_base::out );
  
  virtual ~basic_stringbuf();
  
  __string_type str() const;
  void str(const __string_type& str_arg);
  
 protected:

  virtual int_type overflow(int_type c = traits::eof());
  virtual int_type pbackfail(int_type c = traits::eof());
  virtual int_type underflow();
  virtual pos_type seekoff(off_type off, ios_base::seekdir way,
			   ios_base::openmode which =
			   ios_base::in | ios_base::out);

  virtual pos_type seekpos(pos_type sp,
			   ios_base::openmode which =
			   ios_base::in | ios_base::out);

  virtual basic_streambuf<charT,traits>* setbuf(char_type* s, streamsize n);
  virtual streamsize xsputn(const char_type *s, streamsize n);
#if (G__GNUC>=3)
 private:
  basic_stringbuf& operator=(const basic_stringbuf& x);
#endif
};


template<class charT, class traits, class Allocator>
class basic_istringstream : public basic_istream<charT, traits>
{
 public:
  typedef charT                                           char_type;
  typedef traits::int_type                      int_type;
  typedef traits::pos_type                      pos_type;
  typedef traits::off_type                      off_type;
  typedef traits                                          traits_type;
  
  typedef basic_stringbuf<charT, traits, Allocator>       sb_type;
  typedef basic_ios<charT, traits>                        ios_type;
#ifdef __CINT__
#if 0 && !(G__GNUC>=3)
  typedef string         string_type;
#else
  typedef string         __string_type;
#endif
#else
  typedef basic_string<charT, traits, Allocator >         string_type;
#endif
  
  explicit basic_istringstream(ios_base::openmode which = ios_base::in);
  explicit basic_istringstream(const __string_type& str,
			       ios_base::openmode which = ios_base::in);
#ifdef __CINT__
  explicit basic_istringstream(const char *str,
			       ios_base::openmode which = ios_base::in);
#endif

  virtual ~basic_istringstream();
  
  basic_stringbuf<charT, traits, Allocator> *rdbuf() const;
  __string_type str() const;

  void str(const __string_type& str);
};


template<class charT, class traits, class Allocator>
class basic_ostringstream : public basic_ostream<charT, traits>
{
 public:
  typedef charT                                             char_type;
  typedef traits::int_type                        int_type;
  typedef traits::pos_type                        pos_type;
  typedef traits::off_type                        off_type;
  typedef traits                                            traits_type;
      
  typedef basic_stringbuf<charT, traits, Allocator>         sb_type;
  typedef basic_ios<charT, traits>                          ios_type;
#ifdef __CINT__
  typedef string          __string_type;
#else
  typedef basic_string<charT, traits, Allocator>            string_type;
#endif

  explicit basic_ostringstream(ios_base::openmode which = ios_base::out);
  explicit basic_ostringstream(const __string_type& str,
			       ios_base::openmode which = ios_base::out);

  virtual ~basic_ostringstream();
  basic_stringbuf<charT, traits, Allocator> *rdbuf() const;

  __string_type str() const;
  void str(const __string_type& str);
};


template<class charT, class traits, class Allocator>
class basic_stringstream : public basic_iostream<charT, traits>
{
 public:
  typedef charT                                             char_type;
  typedef traits::int_type                        int_type;
  typedef traits::pos_type                        pos_type;
  typedef traits::off_type                        off_type;
  typedef traits                                            traits_type;
      
  typedef basic_stringbuf<charT, traits, Allocator>         sb_type;
  typedef basic_ios<charT, traits>                          ios_type;
#ifdef __CINT__
  typedef string            __string_type;
#else
  typedef basic_string<charT, traits, Allocator>            string_type;
#endif

  explicit basic_stringstream(ios_base::openmode which = ios_base::out | 
			      ios_base::in);
  
  explicit basic_stringstream(const __string_type& str,
			      ios_base::openmode which = 
			      ios_base::out | ios_base::in);
  
  virtual ~basic_stringstream();
  basic_stringbuf<charT, traits, Allocator> *rdbuf() const;
  __string_type str() const;
  void str(const __string_type& str);
};


//typedef basic_stringbuf<char>    stringbuf;
typedef basic_stringbuf<char,char_traits<char>,allocator<char> > stringbuf;
  
//typedef basic_stringbuf<wchar_t>           wstringbuf;
//typedef basic_stringbuf<wchar_t,char_traits<wchar_t>,allocator<wchar_t> > wstringbuf;

//typedef basic_istringstream<char>      istringstream;
typedef basic_istringstream<char,char_traits<char>,allocator<char> > istringstream;
  
//typedef basic_istringstream<wchar_t>       wistringstream;
//typedef basic_istringstream<wchar_t,char_traits<wchar_t>,allocator<wchar_t> > wistringstream;

//typedef basic_ostringstream<char>    ostringstream;
typedef basic_ostringstream<char,char_traits<char>,allocator<char> > ostringstream;
  
//typedef basic_ostringstream<wchar_t>    wostringstream;
//typedef basic_ostringstream<wchar_t,char_traits<wchar_t>,allocator<wchar_t> > wostringstream;

//typedef basic_stringstream<char>   stringstream;
typedef basic_stringstream<char,char_traits<char>,allocator<char> > stringstream;

//typedef basic_stringstream<wchar_t>  wstringstream;
//typedef basic_stringstream<wchar_t, char_traits<wchar_t>, allocator<wchar_t> > wstringstream;

#endif // __CINT__

#endif // G__SSTREAM_H
