/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file iostrm.h
 ************************************************************************
 * Description:
 *  Stub file for making iostream library for Borland C++ Builder 3.0
 ************************************************************************
 * Copyright(c) 1998-2002   Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__IOSTREAM_H
#define G__IOSTREAM_H

#define G__TMPLTIOS

#define G__OSTREAMBODY
//#define G__OSTREAMMEMBERSTUB
#define G__OSTREAMGLOBALSTUB

#ifndef __CINT__

#include <iostream>
using namespace std;

//inline ostream& operator<< (ostream& ost,unsigned char c) 
//  {return(ost.operator<<(c));}
inline ostream& operator<< (ostream& ost,short c) 
  {return(ost.operator<<(c));}
inline ostream& operator<< (ostream& ost,unsigned short c) 
  {return(ost.operator<<(c));}
inline ostream& operator<< (ostream& ost,int c) 
  {return(ost.operator<<(c));}
inline ostream& operator<< (ostream& ost,unsigned int c) 
  {return(ost.operator<<(c));}
inline ostream& operator<< (ostream& ost,long c) 
  {return(ost.operator<<(c));}
inline ostream& operator<< (ostream& ost,unsigned long c) 
  {return(ost.operator<<(c));}
inline ostream& operator<< (ostream& ost,float c) 
  {return(ost.operator<<(c));}
inline ostream& operator<< (ostream& ost,double c) 
  {return(ost.operator<<(c));}
inline ostream& operator<< (ostream& ost,long double c) 
  {return(ost.operator<<(c));}
inline ostream& operator<< (ostream& ost,bool c) 
  {return(ost.operator<<(c));}
inline ostream& operator<< ( ostream& ost, void* p) 
  {return(ost.operator<<(p));}

//inline istream& operator>> ( istream& ist, unsigned char& p) 
//  {return(ist.operator>>(p));}
inline istream& operator>> ( istream& ist, short& p) 
  {return(ist.operator>>(p));}
inline istream& operator>> ( istream& ist, unsigned short& p) 
  {return(ist.operator>>(p));}
inline istream& operator>> ( istream& ist, int & p) 
  {return(ist.operator>>(p));}
inline istream& operator>> ( istream& ist, unsigned int& p) 
  {return(ist.operator>>(p));}
inline istream& operator>> ( istream& ist, long & p) 
  {return(ist.operator>>(p));}
inline istream& operator>> ( istream& ist, unsigned long& p) 
  {return(ist.operator>>(p));}
inline istream& operator>> ( istream& ist, float & p) 
  {return(ist.operator>>(p));}
inline istream& operator>> ( istream& ist, double & p) 
  {return(ist.operator>>(p));}
inline istream& operator>> ( istream& ist, bool& p) 
  {return(ist.operator>>(p));}
inline istream& operator>> ( istream& ist, long double & p) 
  {return(ist.operator>>(p));}
inline istream& operator>> ( istream& ist, void*& p) 
  {return(ist.operator>>(p));}

#ifndef G__OLDIMPLEMENTATION1635
#include <fstream>
/********************************************************************
 * static variables for iostream redirection
 ********************************************************************/
static ostream::streambuf_type *G__store_cout;
static ostream::streambuf_type *G__store_cerr;
static istream::streambuf_type *G__store_cin;
static ofstream  *G__redirected_cout;
static ofstream  *G__redirected_cerr;
static ifstream  *G__redirected_cin;
/********************************************************************
 * G__redirectcout
 ********************************************************************/
extern "C" void G__unredirectcout() {
  if(G__store_cout) {
    cout.rdbuf(G__store_cout);
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
  G__redirected_cout = new ofstream(filename,ios_base::app);
  G__store_cout = cout.rdbuf(G__redirected_cout->rdbuf()) ;
}
/********************************************************************
 * G__redirectcerr
 ********************************************************************/
extern "C" void G__unredirectcerr() {
  if(G__store_cerr) {
    cerr.rdbuf(G__store_cerr);
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
  G__redirected_cerr = new ofstream(filename,ios_base::app);
  G__store_cerr = cerr.rdbuf(G__redirected_cerr->rdbuf()) ;
}
/********************************************************************
 * G__redirectcin
 ********************************************************************/
extern "C" void G__unredirectcin() {
  if(G__store_cin) {
    cin.rdbuf(G__store_cin);
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
  G__redirected_cin = new ifstream(filename,ios_base::in);
  G__store_cin = cin.rdbuf(G__redirected_cin->rdbuf()) ;
}
#endif /* 1635 */

#else // __CINT__

#include <cstdio>

/********************************************************************
* macro G__MANIP_SUPPORT must be defined to enable true manipulator
*********************************************************************/
#define G__MANIP_SUPPORT

extern "C" {
  typedef struct {
    private:
     int __fill[6];
  } mbstate_t;
}
typedef long streampos ;
typedef long streamoff ;

typedef long         SZ_T;       
typedef SZ_T         streamsize;

class ios_base {
  public:
    typedef int      iostate;
    enum io_state {
	goodbit     = 0x00,   
	badbit      = 0x01,   
	eofbit      = 0x02,  
	failbit     = 0x04  
    };
    typedef int      openmode;
    enum open_mode {
	app         = 0x01,   
	binary      = 0x02,  
	in          = 0x04, 
	out         = 0x08,   
	trunc       = 0x10,                  
	ate         = 0x20 
    };
    typedef int      seekdir;
    enum seek_dir {
	beg         = 0x0,    
	cur         = 0x1,    
	end         = 0x2   
    };        
    typedef int      fmtflags;
    enum fmt_flags {
	boolalpha   = 0x0001,
	dec         = 0x0002,
	fixed       = 0x0004,
	hex         = 0x0008,
	internal    = 0x0010,
	left        = 0x0020,
	oct         = 0x0040,
	right       = 0x0080,
	scientific  = 0x0100,
	showbase    = 0x0200, 
	showpoint   = 0x0400, 
	showpos     = 0x0800, 
	skipws      = 0x1000, 
	unitbuf     = 0x2000, 
	uppercase   = 0x4000, 
	adjustfield = left | right | internal,
	basefield   = dec | oct | hex,
	floatfield  = scientific | fixed
    };
    enum event { 
	erase_event   = 0x0001,
	imbue_event   = 0x0002,
	copyfmt_event = 0x0004
    };
    typedef void (*event_callback) (event, ios_base&, int index);
    void register_callback( event_callback fn, int index);
#ifdef G__SUNCC5
    enum EmptyCtor {emptyctor}; 
#endif
    class Init {
    public:
	static int getinit_cnt_();
	Init();
	~Init();
    };
    inline fmtflags flags() const;
    inline fmtflags flags(fmtflags fmtfl);
    inline fmtflags setf(fmtflags fmtfl);
    inline fmtflags setf(fmtflags fmtfl, fmtflags mask);
    inline void unsetf(fmtflags mask);
    ios_base& copyfmt(const ios_base& rhs);
    inline streamsize precision() const;
    inline streamsize precision(streamsize prec);
    inline streamsize width() const;
    inline streamsize width(streamsize wide);
    static int xalloc();
    long&  iword(int index);
    void*& pword(int index);
    //locale imbue(const locale& loc);
    //locale getloc() const ;
    bool is_synch() ;
#ifdef G__SUNCC5
    static bool sync_with_stdio(bool sync = true);
#else
    bool sync_with_stdio(bool sync = true);
#endif
#ifdef G__SUNCC5
    virtual ~ios_base();    
#endif
  protected:
    ios_base();
#ifndef G__SUNCC5
    ~ios_base();    
#endif
    ios_base& operator=(const ios_base& x);
};

template<class charT, class traits>
class basic_ios : public ios_base { 
  public:
    typedef basic_ios<charT, traits>           ios_type;
    typedef basic_streambuf<charT, traits>     streambuf_type; 
    typedef basic_ostream<charT, traits>       ostream_type;
    typedef traits::char_type      char_type;
    typedef traits                 traits_type;
    typedef traits::int_type       int_type;
    typedef traits::off_type       off_type;
    typedef traits::pos_type       pos_type;
    explicit basic_ios(basic_streambuf<charT, traits> *sb_arg);
    virtual ~basic_ios();
    char_type fill() const;        
    char_type fill(char_type ch);
    inline void exceptions(iostate excpt);
    inline iostate exceptions() const;
    inline void clear(iostate state = goodbit);
    inline void setstate(iostate state);
    inline iostate rdstate() const;
    inline operator void*() const;
    inline bool operator! () const;
    inline bool good() const;
    inline bool eof()  const;
    inline bool fail() const;
    inline bool bad()  const;
    ios_type& copyfmt(const ios_type& rhs);
    inline ostream_type *tie() const;
    ostream_type *tie(ostream_type *tie_arg);
    inline streambuf_type *rdbuf() const;
    streambuf_type *rdbuf( streambuf_type *sb);
    //locale imbue(const locale& loc);
    inline char  narrow(charT, char) const;
    inline charT widen(char) const;
  protected:
    basic_ios();
    void init(basic_streambuf<charT, traits> *sb);
  private:
// #ifdef G__SUNCC5
    basic_ios(const basic_ios& );       //  not defined
    basic_ios& operator=(const basic_ios&);     //  not defined
// #endif
};

template<class charT, class traits>
class basic_streambuf {
  public:
    typedef charT		       	  char_type;
    typedef traits                        traits_type;
    typedef traits::int_type	          int_type;
    typedef traits::pos_type	          pos_type;
    typedef traits::off_type	          off_type;
    virtual ~basic_streambuf();
    //locale pubimbue( const locale& loc);
    //locale getloc() const; 
    inline  basic_streambuf<char_type, traits> *
	pubsetbuf(char_type *s, streamsize n);
    inline pos_type pubseekoff(off_type off, ios_base::seekdir way,
			       ios_base::openmode which =
			       ios_base::in | ios_base::out);
    inline pos_type pubseekpos(pos_type sp, ios_base::openmode which =
			       ios_base::in | ios_base::out);
    inline int pubsync( );
    inline ios_base::openmode which_open_mode();
    inline streamsize   in_avail();
    inline int_type snextc();
    inline int_type sbumpc();
    inline int_type sgetc();
    inline streamsize sgetn(char_type *s, streamsize n);
    inline int_type sputbackc(char_type c);
    inline int_type sungetc();
    inline int_type sputc(char_type c);
    inline streamsize sputn(const char_type *s, streamsize n);
  protected:
    basic_streambuf();
  private:
    basic_streambuf& operator=(const basic_streambuf& x);
};

template<class charT, class traits>
class basic_istream : virtual public basic_ios<charT, traits> {
  public:
    typedef basic_istream<charT, traits>             istream_type;
    typedef basic_ios<charT, traits>                 ios_type;
    typedef basic_streambuf<charT, traits>           streambuf_type;
    typedef traits                      traits_type;
    typedef charT		      	char_type;
    typedef traits::int_type   int_type;
    typedef traits::pos_type   pos_type;
    typedef traits::off_type   off_type;
    explicit basic_istream(basic_streambuf<charT, traits> *sb);
    virtual ~basic_istream();
    class sentry {
    public:
	inline sentry(basic_istream<charT,traits>& stream,bool noskipws = 0);
	~sentry() {}
	operator bool () { return ok_; }
    };
    //istream_type& operator>>(istream_type& (*pf)(istream_type&));
    //istream_type& operator>>(ios_base& (*pf)(ios_base&));
    //istream_type& operator>>(ios_type& (*pf)(ios_type&));
#ifndef __CINT__
    istream_type& operator>>(bool& n);
    istream_type& operator>>(short& n);
    istream_type& operator>>(unsigned short& n);
    istream_type& operator>>(int& n);
    istream_type& operator>>(unsigned int& n);
    istream_type& operator>>(long& n);
    istream_type& operator>>(unsigned long& n);
    istream_type& operator>>(float& f);
    istream_type& operator>>(double& f);
    istream_type& operator>>(long double& f);
    istream_type& operator>>(streambuf_type *sb);
    istream_type& operator>>(void*& p);
#endif
    istream_type& operator>>(streambuf_type& sb);
    int_type get();
    istream_type& get(char_type *s, streamsize n, char_type delim);
    istream_type& get(char_type *s, streamsize n);
    istream_type& get(char_type& c);
    istream_type& get(streambuf_type& sb, char_type delim);
    istream_type& get(streambuf_type& sb);
    istream_type& getline(char_type *s, streamsize n, char_type delim);
    istream_type& getline(char_type *s, streamsize n);
    istream_type& ignore(streamsize n , int_type delim );
    istream_type& ignore(streamsize n =1 );
    //istream_type& ignore(streamsize n = 1, int_type delim = traits::eof());
    istream_type& read(char_type *s, streamsize n);
    streamsize readsome(char_type *s, streamsize n);
    int peek();
    pos_type tellg();
    istream_type& seekg(pos_type pos);
    int sync();
    //#ifndef __CINT__
    istream_type& seekg(off_type, ios_base::seekdir);
    //#endif
    istream_type& putback(char_type c);
    istream_type& unget();
    streamsize gcount() const;
  protected:
    basic_istream( );
};

template<class charT, class traits>
class basic_ostream : virtual public basic_ios<charT, traits> {
  public:
    typedef basic_ostream<charT, traits>           ostream_type;
    typedef basic_ios<charT, traits>               ios_type;
    typedef traits                                 traits_type;
    typedef charT                                  char_type;
    typedef traits::int_type              int_type;
    typedef traits::pos_type              pos_type;
    typedef traits::off_type              off_type;
    explicit basic_ostream(basic_streambuf<charT, traits> *sb);
    virtual ~basic_ostream();
    class sentry {
    public:
      inline explicit sentry(basic_ostream<charT,traits>& stream);
      ~sentry() ;
      operator bool () ;
    private:
//#ifdef G__SUNCC5
      sentry(const sentry&); //   not defined
      sentry& operator=(const sentry&); //   not defined
//#endif
    };
    //ostream_type& operator<<(ostream_type& (*pf)(ostream_type&));
    //ostream_type& operator<<(ios_base& (*pf)(ios_base&));
    //ostream_type& operator<<(ios_type& (*pf)(ios_type&));
#ifndef __CINT__
    ostream_type& operator<<(short n);
    ostream_type& operator<<(unsigned short n);
    ostream_type& operator<<(int n);
    ostream_type& operator<<(unsigned int n);
    ostream_type& operator<<(long n);
    ostream_type& operator<<(unsigned long n);
    ostream_type& operator<<(float f);
    ostream_type& operator<<(double f);
    ostream_type& operator<<(long double f); 
    ostream_type& operator<<(bool n);
    ostream_type& operator<<(basic_streambuf<char_type, traits> *sb);
    ostream_type& operator<<(void *p);
#endif
    //ostream_type& operator<<(basic_streambuf<char_type, traits>& sb);
    ostream_type& put(char_type c);
    ostream_type& write(const char_type *s, streamsize n);
    ostream_type& flush();
    ostream_type& seekp(pos_type pos);
    ostream_type& seekp(off_type , ios_base::seekdir );
    pos_type tellp();
  protected:
    basic_ostream();
};


template<class charT, class traits>
class basic_iostream 
 : public basic_istream<charT,traits>,public basic_ostream<charT,traits> 
{
public:
  explicit basic_iostream(basic_streambuf<charT, traits> *sb);
  virtual ~basic_iostream();
      
protected:
  explicit basic_iostream();
};


typedef int INT_T;

template<class charT>
struct char_traits {
    typedef charT                     char_type;
    typedef INT_T                     int_type;
#ifdef __CINT__
    typedef mbstate_t                 state_type;
    typedef fpos<state_type>         pos_type;
    typedef wstreamoff               off_type;
#endif 
    static void assign (char_type& c1, const char_type& c2)   ;
    static char_type to_char_type(const int_type& c);
    static int_type to_int_type(const char_type& c);
    static bool     eq(const char_type& c1,const char_type& c2);
    static bool lt (const char_type& c1, const char_type& c2) ;
    static int compare (const char_type* s1, const char_type* s2, size_t n);
    static bool     eq_int_type(const int_type& c1,const int_type& c2);
    static int_type             eof();
    static int_type             not_eof(const int_type& c);
    static size_t               length(const char_type *s);
    static const char_type* find (const char_type* s,int n,const char_type& a);
    static char_type  *copy(char_type *dst,const char_type *src, size_t n);
    static char_type* move (char_type* s1, const char_type* s2, size_t n);
    static char_type* assign (char_type* s, size_t n, const char_type a);
};

struct char_traits<char> {
    typedef char                      char_type;
    typedef int                       int_type;
    
#ifdef __CINT__
    typedef streamoff                 off_type; 
    typedef streampos                 pos_type;
    typedef mbstate_t                 state_type;
#endif 
    static void assign (char_type& c1, const char_type& c2)   ;
    static char_type         to_char_type(const int_type& c);
    static int_type          to_int_type(const char_type& c);
    static bool              eq(const char_type& c1,const char_type& c2);
    static bool lt (const char_type& c1, const char_type& c2) ;
    static int compare (const char_type* s1, const char_type* s2, size_t n);
    static const char_type* find (const char_type* s,int n,const char_type& a);
    static bool         eq_int_type(const int_type& c1,const int_type& c2);
    static int_type          eof();
    static int_type          not_eof(const int_type& c);
    static size_t            length(const char_type *s);
    static char_type  *copy(char_type *dst,const char_type *src, size_t n);
    static char_type * move (char_type* s1, const char_type* s2, size_t n);
    static char_type* assign (char_type* s, size_t n, const char_type a);
};

//typedef basic_istream<char> >                   istream;
typedef basic_istream<char, char_traits<char> >   istream;
//typedef basic_ostream<char>                     ostream;
typedef basic_ostream<char, char_traits<char> >   ostream;

extern istream cin ;
extern ostream cout ;
extern ostream cerr ;
extern ostream clog ;

#ifndef G__OLDIMPLEMENTATION1938
ios_base&	dec(ios_base&) ; 
ios_base&	hex(ios_base&) ;
ios_base&	oct(ios_base&) ; 
ios_base&       fixed(ios_base&);
ios_base&       scientific(ios_base&);
ios_base&       right(ios_base&);
ios_base&       left(ios_base&);
ios_base&       internal(ios_base&);
ios_base&       nouppercase(ios_base&);
ios_base&       uppercase(ios_base&);
ios_base&       noskipws(ios_base&);
ios_base&       skipws(ios_base&);
ios_base&       noshowpos(ios_base&);
ios_base&       showpos(ios_base&);
ios_base&       noshowpoint(ios_base&);
ios_base&       showpoint(ios_base&);
ios_base&       noshowbase(ios_base&);
ios_base&       showbase(ios_base&);
ios_base&       noboolalpha(ios_base&);
ios_base&       boolalpha(ios_base&);
#endif

istream&	ws(istream&) ;

ostream&	endl(ostream& i) ;
ostream&	ends(ostream& i) ;
ostream&	flush(ostream&) ;

ostream& operator<< ( ostream&, char );
ostream& operator<< ( ostream&, char* );
ostream& operator<< ( ostream&, void* );
ostream& operator<< ( ostream&, unsigned char );
ostream& operator<< ( ostream&, short );
ostream& operator<< ( ostream&, unsigned short );
ostream& operator<< ( ostream&, int );
ostream& operator<< ( ostream&, unsigned int );
ostream& operator<< ( ostream&, long );
ostream& operator<< ( ostream&, unsigned long);
ostream& operator<< ( ostream&, float );
ostream& operator<< ( ostream&, double );
//ostream& operator<< ( ostream&, long double );
ostream& operator<< ( ostream&, bool );

istream& operator>> ( istream&, char& );
istream& operator>> ( istream&, unsigned char& );
istream& operator>> ( istream&, short& );
istream& operator>> ( istream&, unsigned short& );
istream& operator>> ( istream&, int& );
istream& operator>> ( istream&, unsigned int& );
istream& operator>> ( istream&, long& );
istream& operator>> ( istream&, unsigned long& );
istream& operator>> ( istream&, float& );
istream& operator>> ( istream&, double& );
//istream& operator>> ( istream&, long double& );
istream& operator>> ( istream&, bool& );
istream& operator>> ( istream&, char* );
istream& operator>> ( istream&, void*& );

#endif // __CINT__

#endif // G__IOSTREAM_H
