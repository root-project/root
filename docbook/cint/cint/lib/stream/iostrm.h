/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file iostrm.h
 ************************************************************************
 * Description:
 *  Stub file for making iostream library
 ************************************************************************
 * Copyright(c) 1991~2002,  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__IOSTREAM_H
#define G__IOSTREAM_H

#ifndef __CINT__

#ifdef G__NEWSTDHEADER
#include <iostream>
#else
#include <iostream.h>
#endif

#ifndef G__OLDIMPLEMENTATION1635
#ifdef G__NEWSTDHEADER
#include <fstream>
#else
#include <fstream.h>
#endif
/********************************************************************
 * static variables for iostream redirection
 ********************************************************************/
static streambuf *G__store_cout;
static streambuf *G__store_cerr;
static streambuf *G__store_cin;
static ofstream  *G__redirected_cout;
static ofstream  *G__redirected_cerr;
static ifstream  *G__redirected_cin;
/********************************************************************
 * G__unredirectcout
 ********************************************************************/
extern "C" void G__unredirectcout() {
  if(G__store_cout) {
#if (defined(_AIX) || defined(__sgi) || defined(__alpha) || defined(__hpux)) && !defined(__GNUC__) 
    cout << G__store_cout;
#else
    cout.rdbuf(G__store_cout);
#endif
    G__store_cout = 0;
  }
  if(G__redirected_cout) {
    delete G__redirected_cout;
    G__redirected_cout = 0;
  }
}
/********************************************************************
 * G__redirectcout
 ********************************************************************/
extern "C" void G__redirectcout(const char* filename) {
  G__unredirectcout();
  G__redirected_cout = new ofstream(filename,ios::app);
#if (defined(_AIX) || defined(__sgi) || defined(__alpha) || defined(__hpux)) && !defined(__GNUC__) 
  G__store_cout = cout.rdbuf() ;
  cout << G__redirected_cout->rdbuf() ;
#else
  G__store_cout = cout.rdbuf(G__redirected_cout->rdbuf()) ;
#endif
}
/********************************************************************
 * G__unredirectcerr
 ********************************************************************/
extern "C" void G__unredirectcerr() {
  if(G__store_cerr) {
#if (defined(_AIX) || defined(__sgi) || defined(__alpha) || defined(__hpux)) && !defined(__GNUC__) 
    cerr << G__store_cerr;
#else
    cerr.rdbuf(G__store_cerr);
#endif
    G__store_cerr = 0;
  }
  if(G__redirected_cerr) {
    delete G__redirected_cerr;
    G__redirected_cerr = 0;
  }
}
/********************************************************************
 * G__redirectcerr
 ********************************************************************/
extern "C" void G__redirectcerr(const char* filename) {
  G__unredirectcerr();
  G__redirected_cerr = new ofstream(filename,ios::app);
#if (defined(_AIX) || defined(__sgi) || defined(__alpha) || defined(__hpux)) && !defined(__GNUC__) 
  G__store_cerr = cerr.rdbuf() ;
  cerr << G__redirected_cerr->rdbuf() ;
#else
  G__store_cerr = cerr.rdbuf(G__redirected_cerr->rdbuf()) ;
#endif
}
/********************************************************************
 * G__redirectcin
 ********************************************************************/
extern "C" void G__unredirectcin() {
  if(G__store_cin) {
#if (defined(_AIX) || defined(__sgi) || defined(__alpha) || defined(__hpux)) && !defined(__GNUC__) 
    cin >> G__store_cin;
#else
    cin.rdbuf(G__store_cin);
#endif
    G__store_cin = 0;
  }
  if(G__redirected_cin) {
    delete G__redirected_cin;
    G__redirected_cin = 0;
  }
}
/********************************************************************
 * G__redirectcin
 ********************************************************************/
extern "C" void G__redirectcin(const char* filename) {
  G__unredirectcin();
  G__redirected_cin = new ifstream(filename,ios::in);
#if (defined(_AIX) || defined(__sgi) || defined(__alpha) || defined(__hpux)) && !defined(__GNUC__) 
  G__store_cin = cin.rdbuf() ;
  cin >> G__redirected_cin->rdbuf() ;
#else
  G__store_cin = cin.rdbuf(G__redirected_cin->rdbuf()) ;
#endif
}
#endif /* 1635 */


#else

/********************************************************************
* macro G__MANIP_SUPPORT must be defined to enable true manipulator
*********************************************************************/
#define G__MANIP_SUPPORT

typedef long streampos ;
typedef long streamoff ;

class streambuf ;
class ostream ;

class ios {
 public: /* Some enums are declared in ios to avoid pollution of
	  * global namespace
	  */
  enum io_state	{ goodbit=0, eofbit=1, failbit=2, badbit=4, 
		    hardfail=0200};
  /* hard fail can be set and reset internally,
   * but not via public function */
  enum open_mode	{ in=1, out=2, ate=4, app=010, trunc=020,
			    nocreate=040, noreplace=0100 , binary=0x80 } ;
  enum seek_dir	{ beg=0, cur=1, end=2 } ;
  
  /* flags for controlling format */
  enum		{ skipws=01,	
		    /* skip whitespace on input */
		    left=02,  right=04, internal=010,
		    /* padding location */
		    dec=020, oct=040, hex=0100, 
		    /* conversion base */
		    showbase=0200, showpoint=0400, uppercase=01000,
		    showpos=02000, 
		    /* modifiers */
		    scientific=04000, fixed=010000,
		    /* floating point notation */
		    unitbuf=020000, stdio=040000
		      /* stuff to control flushing */
                    , boolalpha = 100000 , adjustfield = 14
                    , basefield = 112 , floatfield = 6144
		    } ;
  // static const long basefield ; /* dec|oct|hex */
  // static const long adjustfield ; /* left|right|internal */
  // static const long floatfield ; /* scientific|fixed */

 protected:
  ios(streambuf*) ;
  virtual ~ios() ;

 public:
  long		flags() /* const */	{ return x_flags ; }
  long		flags(long f);

  long		setf(long setbits, long field);
  long		setf(long) ;
  long		unsetf(long) ;
  
  int		width() /* const */	{ return x_width ; }
  int		width(int w);
		
  ostream*	tie(ostream* s); 
  ostream*	tie();	
  char		fill(char) ;
  char		fill() /* const */;
  int		precision(int) ;
  int		precision() /* const */;

  int		rdstate() /* const */;
  operator void*();
  // operator const void*() const;
  int	operator!() /* const */; // Alpha/cxx does not like this
  int		eof() /* const */;
  int		fail() /* const */;
  int		bad() /* const */;
  int		good() /* const */;
  void		clear(int i =0) ;
  streambuf*	rdbuf() ;

public: /* Members related to user allocated bits and words */
  long &		iword(int) ;
  void* &		pword(int) ;
  static long	bitalloc() ;
  static int	xalloc() ;

private: /*** privates for implemting allocated bits and words */ 
  static long	nextbit ;
  static long	nextword ;
	
  int		nuser ;
  // union ios_user_union* x_user ;
  void	uresize(int) ;
public: /* static member functions */
  static void	sync_with_stdio() ;
protected:
  /* Does the real work of a constructor */
  ios() ; /* No initialization at all. Needed by
	   * multiple inheritance versions */
  int		assign_private ;
  /* needed by with_assgn classes */
 private:		
  ios(ios&) ; /* Declared but not defined */
  void		operator=(ios&) ; /* Declared but not defined */
 public:   /* old stream package compatibility */
#ifdef G__NEVER
  int		skip(int i) ; 
#endif
};

class streambuf {
 public: 
  virtual int	overflow(int c=EOF);
  virtual int	underflow();
  virtual int	pbackfail(int c);
  virtual int	sync();
  virtual streampos seekoff(streamoff,ios::seek_dir,int =ios::in|ios::out);
  virtual streampos seekpos(streampos, int =ios::in|ios::out) ;
  virtual int	xsputn(const char*  s,int n);
  virtual int	xsgetn(char*  s,int n);

  int		in_avail();
  int		out_waiting() ;
  int		sgetc();
  int		snextc();
  int		sbumpc();
  // int		optim_in_avail();
  // int		optim_sbumpc();
  void		stossc();
  int		sputbackc(char c);
  int		sputc(int c);
  int		sputn(const char*  s,int n);
  int		sgetn(char*  s,int n);
  // virtual streambuf* setbuf(char*  p, int len) ;
  // streambuf*	setbuf(unsigned char*  p, int len) ;
  // streambuf*	setbuf(char*  p, int len, int count) ;
  /* obsolete third argument */
  /*** Constructors -- should be protected ***/
 protected:
  streambuf() ;
 private:
  streambuf(streambuf&) ;
 public:
  // streambuf(char*  p, int l) ;

  // streambuf(char*  p, int l,int c) ; 
  /* 3 argument form is obsolete.
   * Use strstreambuf.
   */
  virtual		~streambuf() ;
};

class istream : virtual public ios {
public: /* Constructor */
  istream(streambuf*) ;
  virtual	~istream() ;
public:	
  int		ipfx(int noskipws /* =0 */);
  void		isfx() { }  
#define G__OLDIMPLEMENTATION762
#ifdef G__OLDIMPLEMENTATION762
  istream&	seekg(streampos p) ;
  istream&	seekg(streamoff o, ios::seek_dir d) ;
#endif
  streampos	tellg() ; 
#ifdef G__MANIP_SUPPORT
  istream&	operator>> (istream& (*f)(istream&));
  istream&	operator>> (ios& (*f)(ios&) ) ;
#endif
  istream&	operator>>(char*);
  istream&	operator>>(unsigned char*);
  istream&	operator>>(unsigned char& c);
  istream&	operator>>(char& c);
  // istream&	rs_complicated(unsigned char& c);
  // istream&	rs_complicated(char& c);
  istream&	operator>>(short&);
  istream&	operator>>(int&);
  istream&	operator>>(long&);
  istream&	operator>>(unsigned short&);
  istream&	operator>>(unsigned int&);
  istream&	operator>>(unsigned long&);
  istream&	operator>>(float&);
  istream&	operator>>(double&);
  istream&	operator>>(streambuf*);
  istream&	get(char* , int lim, char delim='\n');
  // istream&	get(unsigned char* b,int lim, char delim='\n');
  istream&	getline(char* b, int lim, char delim='\n');
  istream&	getline(unsigned char* b, int lim, char delim='\n');
  istream&	get(streambuf& sb, char delim ='\n');
  // istream&	get_complicated(unsigned char& c);
  // istream&	get_complicated(char& c);
  istream&	get(unsigned char& c);
  istream&	get(char& c);
  int 		get();
  int		peek() ;
  istream&	ignore(int n=1,int delim=EOF) ;
  istream&	read(char*  s,int n);
  // istream&	read(unsigned char* s,int n) ;
  int		gcount() ;
  istream&	putback(char c);
  // int		sync();

protected:  
  istream() ;

public: /*** Obsolete constructors, carried over from stream package ***/
#ifdef G__NEVER
  istream(streambuf*, int sk, ostream* t=0) ; /* obsolete, set sk and tie
					       * via format state variables */
  istream(int size ,char*,int sk=1) ; /* obsolete, use strstream */
  istream(int fd,int sk=1, ostream* t=0) ; /* obsolete use fstream */
#endif
};

class ostream : virtual public ios {
public: /* Constructor */
  ostream(streambuf*) ;
  virtual	~ostream();
public:	
  int		opfx();	/* Output prefix */
  void		osfx() ;

  ostream&	flush() ;
#ifdef G__OLDIMPLEMENTATION762
  ostream&	seekp(streampos p) ;
  ostream&	seekp(streamoff o, ios::seek_dir d) ;
#endif
  streampos	tellp() ; 
  ostream&	put(char c);
  // ostream&	complicated_put(char c);
  ostream&	operator<<(char c);
  ostream&	operator<<(unsigned char c) ;
  // ostream& 	ls_complicated(char);
  // ostream& 	ls_complicated(unsigned char);

  ostream&	operator<<(const char*);
  ostream&	operator<<(int a); 
  ostream&	operator<<(long);	
  ostream&	operator<<(double);
  ostream&	operator<<(float);
  ostream&	operator<<(unsigned int a);
  ostream&	operator<<(unsigned long);
  ostream&	operator<<(void*);
  /*	ostream&	operator<<(const void*);   add this later */
  ostream&	operator<<(streambuf*);
  ostream&	operator<<(short i); 
  ostream&	operator<<(unsigned short i) ;

#ifdef G__MANIP_SUPPORT
  ostream&	operator<< (ostream& (*f)(ostream&));
  ostream&	operator<< (ios& (*f)(ios&) ) ;
#endif

  ostream&	write(const char*  s,int n)	;
  // ostream&	write(const unsigned char* s, int n);

public: /*** Obsolete constructors, carried over from stream package ***/
#ifdef G__NEVER
  ostream(int fd) ; /* obsolete use fstream */
  ostream(int size ,char*) ; /* obsolete, use strstream */
#endif
} ;

class iostream : public istream, public ostream {
public:
  iostream(streambuf*) ;
  virtual		~iostream() ;
protected:
  iostream() ;
} ;


extern istream cin ;
extern ostream cout ;
extern ostream cerr ;
extern ostream clog ;


ios&		dec(ios&) ; 
ostream&	endl(ostream& i) ;
ostream&	ends(ostream& i) ;
ostream&	flush(ostream&) ;
ios&		hex(ios&) ;
ios&		oct(ios&) ; 
istream&	ws(istream&) ;


#endif /* __CINT__ */



#endif
