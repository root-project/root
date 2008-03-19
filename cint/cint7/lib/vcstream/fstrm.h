/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file fstrm.h
 ************************************************************************
 * Description:
 *  Stub file for making iostream library
 ************************************************************************
 * Copyright(c) 1996       Osamu Kotanigawa
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__FSTREAM_H
#define G__FSTREAM_H

#ifndef __CINT__

#include <fstream.h>

#else

#include "iostrm.h"

class  filebuf : public streambuf {	
  /* a stream buffer for files */
public:
  static const int openprot ; /* default protection for open */
public:
  filebuf() ;
  filebuf(int fd);
  filebuf(int fd, char*  p, int l) ;
  
  int		is_open();
  int		fd();
  filebuf*	open(const char *name, int om, int prot=openprot);
  filebuf*	attach(int fd) ;
  // int		detach() ;
  filebuf* 	close() ;
  ~filebuf() ;
 public: /* virtuals */
  virtual int	overflow(int=EOF);
  virtual int	underflow();
  virtual int	sync() ;
  virtual streampos seekoff(streamoff,ios::seek_dir,int) ;
  virtual streambuf* setbuf(char*  p, int len) ;
};

class ifstream : public istream {
public:
  ifstream() ;
  ifstream(const char* name, int mode=ios::in, int prot=filebuf::openprot) ;
  ifstream(int fd) ;
  ifstream(int fd, char*  p, int l) ;
  ~ifstream() ;

  streambuf * setbuf(char *, int);
  filebuf* rdbuf() const ;

  void attach(int);
  int fd() const ;

  int is_open() const ;
  void open(const char *, int =ios::in, int = filebuf::openprot);
  void close();
//  int setmode(int mode = filebuf::text) ;
} ;

class ofstream : public ostream {
public:
  ofstream() ;
  ofstream(const char* name, int mode=ios::out, int prot=filebuf::openprot) ;
  ofstream(int fd) ;
  ofstream(int fd, char*  p, int l) ;
  ~ofstream() ;

  streambuf * setbuf(char *, int);
  filebuf* rdbuf() const ;

  void attach(int);
  int fd() const ;

  int is_open() const ;
  void open(const char *, int =ios::out, int = filebuf::openprot);
  void close();
//  int setmode(int mode = filebuf::text) ;
} ;

class fstream : public iostream {
public:
  fstream() ;
	
  fstream(const char* name, int mode, int prot=filebuf::openprot) ;
  fstream(int fd) ;
  fstream(int fd, char*  p, int l) ;
  ~fstream() ;

  streambuf * setbuf(char *, int);
  filebuf* rdbuf() const ;

  void attach(int);
  int fd() const ;

  int is_open() const ;
  void open(const char *, int, int = filebuf::openprot);
  void close();
//  int setmode(int mode = filebuf::text) ;
} ;

#endif
#endif

