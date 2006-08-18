/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file fstrm.h
 ************************************************************************
 * Description:
 *  Stub file for making iostream library
 ************************************************************************
 * Copyright(c) 1999       Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__FSTREAM_H
#define G__FSTREAM_H

#ifndef __CINT__

#include <fstream>
using namespace std;

#else // __CINT__

#include "iostrm.h"

template<class charT, class traits>
class basic_filebuf : public basic_streambuf<charT, traits> {
  private:
    typedef basic_ios<charT, traits>              ios_type;
#ifndef __CINT__
    typedef traits::state_type           state_t; 
#endif
  public:
    typedef traits              traits_type;
    typedef charT		char_type;
    typedef traits::int_type    int_type;
    typedef traits::pos_type    pos_type;
    typedef traits::off_type    off_type;
    basic_filebuf();
    virtual ~basic_filebuf();
    bool is_open() const;
    basic_filebuf<charT, traits> * open(const char *s, ios_base::openmode);
    basic_filebuf<charT, traits> *close();
  protected:
    virtual int      showmanyc();
    virtual int_type overflow(int_type c = traits::eof());
    virtual int_type pbackfail(int_type c = traits::eof());
    virtual int_type underflow();
    virtual basic_streambuf<charT,traits>* setbuf(char_type *s,streamsize n);
    virtual pos_type seekoff(off_type off,ios_base::seekdir way
                             ,ios_base::openmode which =
                                       ios_base::in | ios_base::out);
    virtual pos_type seekpos(pos_type sp
                             ,ios_base::openmode which =
                                       ios_base::in | ios_base::out);
    virtual int sync();
    virtual streamsize xsputn(const char_type *s, streamsize n);
 private:
    basic_filebuf& operator=(const basic_filebuf& x);
};

template<class charT, class traits>
class basic_ifstream : public basic_istream<charT, traits> {
  public:
    typedef basic_ios<charT, traits>          ios_type;
    typedef traits                            traits_type;
    typedef charT		       	      char_type;
    typedef traits::int_type                  int_type;
    typedef traits::pos_type                  pos_type;
    typedef traits::off_type                  off_type;
  public:
    basic_ifstream();
    basic_ifstream(const char *s,ios_base::openmode mode = ios_base::in);
    virtual ~basic_ifstream();
    basic_filebuf<charT, traits> *rdbuf() const;
    bool is_open();
    void open(const char *s, ios_base::openmode mode = ios_base::in);
    void close();
  private:
    basic_ifstream(const basic_ifstream& x);  // not implemented
    basic_ifstream& operator=(const basic_ifstream& x); // not implemented
};

template<class charT, class traits>
class basic_ofstream : public basic_ostream<charT, traits> {
  public:
    typedef basic_ios<charT, traits>          ios_type;
    typedef traits                            traits_type;
    typedef charT		              char_type;
    typedef traits::int_type                  int_type;
    typedef traits::pos_type                  pos_type;
    typedef traits::off_type                  off_type;
  public:
    basic_ofstream();
    basic_ofstream(const char *s, ios_base::openmode mode=ios_base::out);
    virtual ~basic_ofstream();
    basic_filebuf<charT, traits> *rdbuf() const;
    bool is_open();
    void open(const char *s,ios_base::openmode mode=ios_base::out);
    void close();
  private:
    basic_ofstream(const basic_ofstream& x);  // not implemented
    basic_ofstream& operator=(const basic_ofstream& x); // not implemented
 };

template<class charT, class traits>
class basic_fstream : public basic_iostream<charT, traits> {
 public:
    basic_fstream();
    basic_fstream(const char *s,ios_base::openmode mode);
    basic_filebuf<charT, traits> *rdbuf() const;
    bool is_open();
    void open(const char *s,ios_base::openmode mode);
    void close();
  private:
    basic_fstream(const basic_fstream& x);  // not implemented
    basic_fstream& operator=(const basic_fstream& x); // not implemented
};

typedef basic_filebuf<char, char_traits<char> >         filebuf;
//typedef basic_filebuf<wchar_t, char_traits<wchar_t> >   wfilebuf;
typedef basic_ifstream<char, char_traits<char> >        ifstream;
//typedef basic_ifstream<wchar_t, char_traits<wchar_t> >  wifstream;
typedef basic_ofstream<char, char_traits<char> >        ofstream;
//typedef basic_ofstream<wchar_t, char_traits<wchar_t> >  wofstream;
typedef basic_fstream<char, char_traits<char> >        fstream;
//typedef basic_fstream<wchar_t, char_traits<wchar_t> >  wfstream;


#endif // __CINT__
#endif // G__FSTREAM_H
