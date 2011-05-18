/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * I/O stream header file iostream.h
 ************************************************************************
 * Description:
 *  CINT iostream header file
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__IOSTREAM_H

/*********************************************************************
* Try initializaing precompiled iostream library
*********************************************************************/
#pragma setstream
#pragma ifdef G__IOSTREAM_H

#pragma ifdef G__TMPLTIOS
typedef ios_base ios;
#pragma else // G__TMPLTIOS
typedef ios ios_base;
#pragma endif // G__TMPLTIOS

#pragma ifndef G__KCC
#pragma include <iosenum.h>

#pragma ifndef G__SSTREAM_H
typedef ostrstream ostringstream;
typedef istrstream istringstream;
//typedef strstream stringstream;  // problem, 
#pragma else // G__SSTREAM_H
typedef ostringstream ostrstream;
typedef istringstream istrstream;
typedef stringstream strstream;
#pragma endif // G__SSTREAM_H

#pragma endif // G__KCC

#pragma endif // G__IOSTREAM_H

#include <bool.h>

/*********************************************************************
* Use fake iostream only if precompiled version does not exist.
*********************************************************************/
#pragma if !defined(G__IOSTREAM_H) // && !defined(__cplusplus)
#define G__IOSTREAM_H

#pragma security level0

#include <stdio.h>

/*********************************************************************
* ios
*
*********************************************************************/
typedef long streamoff;
typedef long streampos;
//class io_state;
class streambuf;
class fstreambase;
typedef long         SZ_T;       
typedef SZ_T         streamsize;

class ios {
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
  
  ios() { x_width=0; }
  streamsize width(streamsize wide) { x_width=wide; }
 protected:
  int x_width;
};


/*********************************************************************
* ostream
*
*********************************************************************/

class ostream : /* virtual */ public ios {
	FILE *fout;
      public:
	ostream(FILE *setfout) { fout=setfout; }
	ostream(char *fname) ;
	~ostream() ;
	void close() { if(fout) fclose(fout); fout=NULL;}
	void flush() { if(fout) fflush(fout); }
	FILE *fp() { return(fout); }
	int rdstate() ;

	ostream& operator <<(char c);
	ostream& operator <<(char *s);
	ostream& operator <<(long i);
	ostream& operator <<(unsigned long i);
	ostream& operator <<(double d);
	ostream& operator <<(void *p);
	ostream& form(char *format ...);
};

ostream::~ostream()
{
  if(fout!=stderr && fout!=stdout && fout!=NULL) {
    fclose(fout);
  }
}

ostream::ostream(char *fname)
{
  fout = fopen(fname,"w");
  if(fout==NULL) {
    fprintf(stderr,"%s can not open\n",fname);
  }
}

ostream& ostream::operator <<(char c)
{
  if(x_width) {
    int init=0;
    if(isprint(c)) init=1;
    for(int i=init;i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  fprintf(fout,"%c",c);
  return(*this);
}

ostream& ostream::operator <<(char *s)
{
  if(x_width &&(!s || x_width>strlen(s))) {
    if(s) for(int i=strlen(s);i<x_width;i++) fputc(' ',fout);
    else  for(int i=0;i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  fprintf(fout,"%s",s);
  return(*this);
}

ostream& ostream::operator <<(long x)
{
  if(x_width) {
    char buf[50];
    sprintf(buf,"%d",x);
    if(x_width>strlen(buf)) 
      for(int i=strlen(buf);i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  fprintf(fout,"%d",x);
  return(*this);
}

ostream& ostream::operator <<(unsigned long x)
{
  if(x_width) {
    char buf[50];
    sprintf(buf,"%u",x);
    if(x_width>strlen(buf)) 
      for(int i=strlen(buf);i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  fprintf(fout,"%u",x);
  return(*this);
}

ostream& ostream::operator <<(double d)
{
  if(x_width) {
    char buf[50];
    sprintf(buf,"%g",d);
    if(x_width>strlen(buf)) 
      for(int i=strlen(buf);i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  fprintf(fout,"%g",d);
  return(*this);
}

ostream& ostream::operator <<(void *p)
{
  if(x_width) {
    char buf[50];
    sprintf(buf,"0x%x",p);
    if(x_width>strlen(buf)) 
      for(int i=strlen(buf);i<x_width;i++) fputc(' ',fout);
    x_width=0;
  }
  printf("0x%x",p);
  return(*this);
}

int ostream::rdstate()
{
  if(fout) return(0);
  else   return(1);
}

/* instanciation of cout,cerr */
ostream cout=ostream(stdout);
ostream cerr=ostream(stderr);


/*********************************************************************
* istream
*
*********************************************************************/

class istream : /* virtual */ public ios {
  FILE *fin;
  ostream *tie;
public:
  istream(FILE *setfin) { fin = setfin; tie=(ostream*)NULL; }
  istream(char *fname);
  ~istream();
  void close() { if(fin) fclose(fin); fin=NULL;}
  ostream& tie(ostream& cx); 
  FILE *fp() { return(fin); }
  int rdstate();
  
  istream& operator >>(char& c);
  istream& operator >>(char *s);
  istream& operator >>(short& s);
  istream& operator >>(int& i);
  istream& operator >>(long& i);
  istream& operator >>(unsigned char& c);
  istream& operator >>(unsigned short& s);
  istream& operator >>(unsigned int& i);
  istream& operator >>(unsigned long& i);
  istream& operator >>(double& d);
  istream& operator >>(float& d);
};

istream::~istream()
{
  if(fin!=stdin && fin!=NULL) {
    fclose(fin);
  }
}

istream::istream(char *fname)
{
  fin = fopen(fname,"r");
  if(fin==NULL) {
    fprintf(stderr,"%s can not open\n",fname);
  }
  tie=(ostream*)NULL;
}


ostream& istream::tie(ostream& cx) 
     
{ 
  ostream *tmp; 
  tmp=tie; 
  tie = &cx; 
  return(*tmp);
}

istream& istream::operator >>(char& c)
{
  if(tie) tie->flush();
  c=fgetc(fin);
  return(*this);
}

istream& istream::operator >>(char *s)
{
  if(tie) tie->flush();
  fscanf(fin,"%s",s);
  return(*this);
}

istream& istream::operator >>(short& s)
{
  if(tie) tie->flush();
  fscanf(fin,"%hd",&s);
  return(*this);
}

istream& istream::operator >>(int& i)
{
  if(tie) tie->flush();
  fscanf(fin,"%d",&i);
  return(*this);
}

istream& istream::operator >>(long& i)
{
  if(tie) tie->flush();
  fscanf(fin,"%ld",&i);
  return(*this);
}

istream& istream::operator >>(unsigned char& c)
{
  int i;
  if(tie) tie->flush();
  fscanf(fin,"%u",&i);
  c = i;
  return(*this);
}
istream& istream::operator >>(unsigned short& s)
{
  if(tie) tie->flush();
  fscanf(fin,"%hu",&s);
  return(*this);
}
istream& istream::operator >>(unsigned int& i)
{
  if(tie) tie->flush();
  fscanf(fin,"%u",&i);
  return(*this);
}
istream& istream::operator >>(unsigned long& i)
{
  if(tie) tie->flush();
  fscanf(fin,"%lu",&i);
  return(*this);
}

istream& istream::operator >>(float& f)
{
  if(tie) tie->flush();
  fscanf(fin,"%g",&f);
  return(*this);
}

istream& istream::operator >>(double& d)
{
  if(tie) tie->flush();
  fscanf(fin,"%lg",&d);
  return(*this);
}

int istream::rdstate()
{
  int cx;
  if(!fin) return(1);
  cx = fgetc(fin);
  fseek(fin,-1,SEEK_CUR);
  if(EOF==cx) return(1);
  return(0);
}

/* instanciation of cin */
istream cin=istream(stdin);

/*********************************************************************
* iostream
*
*********************************************************************/
class iostream : public istream , public ostream {
 public:
  iostream(FILE *setfin) : istream(setfin), ostream(setfin) { }
  iostream(char *fname) : istream(fname), ostream(fname) { }
};


/*********************************************************************
* ofstream, ifstream 
*
*********************************************************************/

class fstream;

class ofstream : public ostream {
 public:
  ofstream(FILE* setfin) : ostream(setfin) { }
  ofstream(char* fname) : ostream(fname) { }
};

class ifstream : public istream {
 public:
  ifstream(FILE* setfin) : istream(setfin) { }
  ifstream(char* fname) : istream(fname) { }
};

class iofstream : public iostream {
 public:
  iofstream(FILE* setfin) : iostream(setfin) { }
  iofstream(char* fname) : iostream(fname) { }
};


ostream& flush(ostream& i) {i.flush(); return(i);}
ostream& endl(ostream& i) {return i << '\n' << flush;}
ostream& ends(ostream& i) {return i << '\0';}
istream& ws(istream& i) {
  fprintf(stderr,"Limitation: ws,WS manipurator not supported\n");
  return(i);
}
istream& WS(istream& i) {
  fprintf(stderr,"Limitation: ws,WS manipurator not supported\n");
  return(i);
}

#pragma endif /* G__IOSTREAM_H */

ostream& ostream::form(char *format ...) {
  char temp[1024];
  return(*this<<G__charformatter(0,temp,1024));
}

/*********************************************************************
* iostream manipurator emulation
*
*  Following description must be deleted when pointer to compiled 
* function is fully supported.
*********************************************************************/
struct G__CINT_IOFLAGS {
   G__CINT_IOFLAGS(int f = 0, int m = 0): flag(f), mask(m) {}
#pragma ifndef G__TMPLTIOS
   typedef ios ios_base;
#pragma endif
   int flag, mask;
};
class G__CINT_ENDL { int dmy; } endl;
class G__CINT_ENDS { int dmy; } ends;
class G__CINT_FLUSH { int dmy; } flush;
class G__CINT_ws { int dmy; } ws;
class G__CINT_WS { int dmy; } WS;

#define G__DECL_IOFM(WHAT, MASK) \
   G__CINT_IOFLAGS WHAT(ios_base::WHAT, ios_base::MASK);
G__DECL_IOFM(hex,basefield);
G__DECL_IOFM(oct,basefield);
G__DECL_IOFM(dec,basefield);
G__DECL_IOFM(scientific,floatfield);
G__DECL_IOFM(fixed,floatfield);
/*
Better not, or "left" will become a CINT reserved variable.
G__DECL_IOFM(left,adjustfield);
G__DECL_IOFM(right,adjustfield);
G__DECL_IOFM(internal,adjustfield);
*/
#undef G__DECL_IOFM

#define G__DECL_IOF(WHAT)\
   G__CINT_IOFLAGS WHAT(ios_base::WHAT, ios_base::WHAT);
G__DECL_IOF(boolalpha);
G__DECL_IOF(showbase);
G__DECL_IOF(showpoint);
G__DECL_IOF(showpos);
G__DECL_IOF(skipws);
G__DECL_IOF(unitbuf);
G__DECL_IOF(uppercase);


/*
class G__CINT_HEX { int dmy; } hex;
class G__CINT_DEC { int dmy; } dec;
class G__CINT_OCT { int dmy; } oct;
class G__CINT_OCT { int dmy; } showpoint;
class G__CINT_SCIENTIFIC { int dmy; } scientific;
class G__CINT_FIXED { int dmy; } fixed;
*/
class G__CINT_NOSUPPORT { int dmy; } ;

#ifndef G__STD_IOSTREAM
#include <_iostream>
#endif

#endif
