/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file fstream.sut.h
 ************************************************************************
 * Description:
 *  Stub file for making iostream library
 ************************************************************************
 * Copyright(c) 1991~1999,  Masaharu Goto (MXJ02154@niftyserve.or.jp)
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

class fstreambase : virtual public ios { 
public:
  fstreambase() ;
	
  fstreambase(const char* name, int mode, int prot=filebuf::openprot) ;
  fstreambase(int fd) ;
  fstreambase(int fd, char*  p, int l) ;
  ~fstreambase() ;
  void open(const char* name, int mode, int prot=filebuf::openprot) ;
  // void		attach(int fd);
  // int		detach();
  void		close() ;
  void		setbuf(char*  p, int l) ;
  filebuf*	rdbuf() { return &buf ; }
} ;

class ifstream : public fstreambase, public istream {
public:
  ifstream() ;
  ifstream(const char* name, int mode=ios::in, int prot=filebuf::openprot) ;
  ifstream(int fd) ;
  ifstream(int fd, char*  p, int l) ;
  ~ifstream() ;

  filebuf*	rdbuf() { return fstreambase::rdbuf(); }
  void		open(const char* name, int mode=ios::in, 
					int prot=filebuf::openprot) ;
} ;

class ofstream : public fstreambase, public ostream {
public:
  ofstream() ;
  ofstream(const char* name, int mode=ios::out, int prot=filebuf::openprot) ;
  ofstream(int fd) ;
  ofstream(int fd, char*  p, int l) ;
  ~ofstream() ;

  filebuf*	rdbuf() { return fstreambase::rdbuf(); }
  void open(const char* name, int mode=ios::out, int prot=filebuf::openprot) ;
} ;

class fstream : public fstreambase, public iostream {
public:
  fstream() ;
	
  fstream(const char* name, int mode, int prot=filebuf::openprot) ;
  fstream(int fd) ;
  fstream(int fd, char*  p, int l) ;
  ~fstream() ;
  filebuf*	rdbuf() { return fstreambase::rdbuf(); }
  void open(const char* name, int mode, int prot=filebuf::openprot) ;
} ;

#endif
#endif
