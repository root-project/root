# 1 "Simple.cxx"
//
// Simple.cxx
//
# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h" 1
/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * I/O stream header file iostream.h
 ************************************************************************
 * Description:
 *  CINT iostream header file
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/



/*********************************************************************
* Try initializaing precompiled iostream library
*********************************************************************/
#pragma setstream
#pragma ifdef G__IOSTREAM_H
#pragma ifndef G__KCC
#pragma ifndef G__TMPLTIOS
#pragma include <iosenum.h>
#pragma endif
#pragma ifndef G__SSTREAM_H
typedef ostrstream ostringstream;
typedef istrstream istringstream;
//typedef strstream stringstream;
#pragma else
typedef ostringstream ostrstream;
typedef istringstream istrstream;
#pragma endif
#pragma endif
#pragma endif

# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/bool.h" 1
#pragma ifndef G__BOOL_H
#pragma define G__BOOL_H

#pragma ifdef G__OLDIMPLEMENTATION1604
/* This header file may not be needed any more */

//#undef FALSE
//#undef TRUE



// bool as fundamental type
const bool false=0,true=1;



bool bool() { return false; }

// This is not needed due to fix 1584
//#pragma link off class bool;
//#pragma link off function bool;

#pragma endif

#pragma endif
# 44 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h" 2

/*********************************************************************
* Use fake iostream only if precompiled version does not exist.
*********************************************************************/
#pragma if !defined(G__IOSTREAM_H)


#pragma security level0

# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/stdio.h" 1



#pragma setstdio

typedef long fpos_t;
typedef unsigned int size_t;
# 25 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/stdio.h"
# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/bool.h" 1
#pragma ifndef G__BOOL_H
#pragma define G__BOOL_H

#pragma ifdef G__OLDIMPLEMENTATION1604
/* This header file may not be needed any more */

//#undef FALSE
//#undef TRUE



// bool as fundamental type
const bool false=0,true=1;



bool bool() { return false; }

// This is not needed due to fix 1584
//#pragma link off class bool;
//#pragma link off function bool;

#pragma endif

#pragma endif
# 26 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/stdio.h" 2

#pragma include_noerr <stdfunc.dll>
# 54 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h" 2

/*********************************************************************
* ios
*
*********************************************************************/
typedef long streamoff;
typedef long streampos;
//class io_state;
class streambuf;
class fstreambase;
typedef long SZ_T;
typedef SZ_T streamsize;

class ios {
 public:
  typedef int iostate;
  enum io_state {
    goodbit = 0x00,
    badbit = 0x01,
    eofbit = 0x02,
    failbit = 0x04
  };
  typedef int openmode;
  enum open_mode {
    app = 0x01,
    binary = 0x02,
    in = 0x04,
    out = 0x08,
    trunc = 0x10,
    ate = 0x20
  };
  typedef int seekdir;
  enum seek_dir {
    beg = 0x0,
    cur = 0x1,
    end = 0x2
  };
  typedef int fmtflags;
  enum fmt_flags {
    boolalpha = 0x0001,
    dec = 0x0002,
    fixed = 0x0004,
    hex = 0x0008,
    internal = 0x0010,
    left = 0x0020,
    oct = 0x0040,
    right = 0x0080,
    scientific = 0x0100,
    showbase = 0x0200,
    showpoint = 0x0400,
    showpos = 0x0800,
    skipws = 0x1000,
    unitbuf = 0x2000,
    uppercase = 0x4000,
    adjustfield = left | right | internal,
    basefield = dec | oct | hex,
    floatfield = scientific | fixed
  };
  enum event {
    erase_event = 0x0001,
    imbue_event = 0x0002,
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
    else for(int i=0;i<x_width;i++) fputc(' ',fout);
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
  else return(1);
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
  fseek(fin,-1,(1));
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

#pragma endif

ostream& ostream::form(char *format ...) {
  char temp[1024];
  return(*this<<G__charformatter(0,temp));
}

/*********************************************************************
* iostream manipurator emulation
*
*  Following description must be deleted when pointer to compiled 
* function is fully supported.
*********************************************************************/
class G__CINT_ENDL { int dmy; } endl;
class G__CINT_ENDS { int dmy; } ends;
class G__CINT_FLUSH { int dmy; } flush;
class G__CINT_ws { int dmy; } ws;
class G__CINT_WS { int dmy; } WS;
class G__CINT_HEX { int dmy; } hex;
class G__CINT_DEC { int dmy; } dec;
class G__CINT_OCT { int dmy; } oct;
class G__CINT_NOSUPPORT { int dmy; } ;


# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/_iostream" 1
// include/_iostream

std::ostream& operator<<(std::ostream& ostr,std::G__CINT_ENDL& i)
        {return(std::endl(ostr));}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_ENDS& i)
        {return(std::ends(ostr));}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_FLUSH& i)
        {return(std::flush(ostr));}
std::istream& operator>>(std::istream& istr,std::G__CINT_ws& i)
        {return(std::ws(istr));}
std::istream& operator>>(std::istream& istr,std::G__CINT_WS& i)
        {return(std::WS(istr));}


std::ostream& operator<<(std::ostream& ostr,std::G__CINT_HEX& i) {
  ostr.unsetf(ios::dec);
  ostr.unsetf(ios::oct);
  ostr.setf(ios::hex);
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_HEX& i) {
  istr.unsetf(ios::dec);
  istr.unsetf(ios::oct);
  istr.setf(ios::hex);
  return(istr);
}

std::ostream& operator<<(std::ostream& ostr,std::G__CINT_DEC& i) {
  ostr.unsetf(ios::hex);
  ostr.unsetf(ios::oct);
  ostr.setf(ios::dec);
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_DEC& i) {
  istr.unsetf(ios::hex);
  istr.unsetf(ios::oct);
  istr.setf(ios::dec);
  return(istr);
}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_OCT& i) {
  ostr.unsetf(ios::hex);
  ostr.unsetf(ios::dec);
  ostr.setf(ios::oct);
  return(ostr);
}
std::istream& operator>>(std::istream& istr,std::G__CINT_OCT& i) {
  istr.unsetf(ios::hex);
  istr.unsetf(ios::dec);
  istr.setf(ios::oct);
  return(istr);
}
std::ostream& operator<<(std::ostream& ostr,std::G__CINT_NOSUPPORT& i) {
  fprintf(stderr,"Limitation: dec,hex,oct manipurator not supported\n");
  return(ostr);
}
std::istream& operator<<(std::istream& istr,std::G__CINT_NOSUPPORT& i) {
  fprintf(stderr,"Limitation: dec,hex,oct manipurator not supported\n");
  return(istr);
}

// Value evaluation
//template<class T> int G__ateval(const T* x) {return(0);}
template<class T> int G__ateval(const T& x) {return(0);}
int G__ateval(const char* x) {return(0);}
int G__ateval(const void* x) {return(0);}
int G__ateval(const double x) {return(0);}
int G__ateval(const float x) {return(0);}
int G__ateval(const char x) {return(0);}
int G__ateval(const short x) {return(0);}
int G__ateval(const int x) {return(0);}
int G__ateval(const long x) {return(0);}
int G__ateval(const unsigned char x) {return(0);}
int G__ateval(const unsigned short x) {return(0);}
int G__ateval(const unsigned int x) {return(0);}
int G__ateval(const unsigned long x) {return(0);}
# 470 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h" 2
# 5 "Simple.cxx" 2
# 1 "Simple.h" 1


//
//
// Simple class
//
# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TShape.h" 1
// @(#)root/g3d:$Name:  $:$Id: TShape.h,v 1.3 2000/12/13 15:13:47 brun Exp $
// Author: Nenad Buncic   17/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TShape                                                               //
//                                                                      //
// Basic shape class                                                    //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TNamed.h" 1
// @(#)root/base:$Name:  $:$Id: TNamed.h,v 1.4 2001/02/13 07:54:00 brun Exp $
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNamed                                                               //
//                                                                      //
// The basis for a named object (name, title).                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////



# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TObject.h" 1
// @(#)root/base:$Name:  $:$Id: TObject.h,v 1.19 2002/04/08 15:06:08 rdm Exp $
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObject                                                              //
//                                                                      //
// Mother of all ROOT objects.                                          //
//                                                                      //
// The TObject class provides default behaviour and protocol for all    //
// objects in the ROOT system. It provides protocol for object I/O,     //
// error handling, sorting, inspection, printing, drawing, etc.         //
// Every object which inherits from TObject can be stored in the        //
// ROOT collection classes.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/Rtypes.h" 1
/* @(#)root/base:$Name:  $:$Id: Rtypes.h,v 1.15 2002/02/26 11:11:19 brun Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/




//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Rtypes                                                               //
//                                                                      //
// Basic types used by ROOT.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////



# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/RConfig.h" 1
/* @(#)root/base:$Name:  $:$Id: RConfig.h,v 1.33 2002/04/11 18:16:16 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/




/*************************************************************************
 *                                                                       *
 * RConfig                                                               *
 *                                                                       *
 * Defines used by ROOT.                                                 *
 *                                                                       *
 *************************************************************************/


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/RVersion.h" 1



/* Version information automatically generated by installer. */

/*
 * These macros can be used in the following way:
 *
 *    #if ROOT_VERSION_CODE >= ROOT_VERSION(2,23,4)
 *       #include <newheader.h>
 *    #else
 *       #include <oldheader.h>
 *    #endif
 *
*/
# 24 "/cdf/home/pcanal/scratch/code/root.merging/include/RConfig.h" 2



/*---- new C++ features ------------------------------------------------------*/




/*---- machines --------------------------------------------------------------*/
# 126 "/cdf/home/pcanal/scratch/code/root.merging/include/RConfig.h"
# 1 "/usr/include/features.h" 1 3
/* Copyright (C) 1991,92,93,95,96,97,98,99,2000,2001 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, write to the Free
   Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307 USA.  */




/* These are defined by the user (or the compiler)
   to specify the desired environment:

   __STRICT_ANSI__	ISO Standard C.
   _ISOC99_SOURCE	Extensions to ISO C89 from ISO C99.
   _POSIX_SOURCE	IEEE Std 1003.1.
   _POSIX_C_SOURCE	If ==1, like _POSIX_SOURCE; if >=2 add IEEE Std 1003.2;
			if >=199309L, add IEEE Std 1003.1b-1993;
			if >=199506L, add IEEE Std 1003.1c-1995
   _XOPEN_SOURCE	Includes POSIX and XPG things.  Set to 500 if
			Single Unix conformance is wanted, to 600 for the
			upcoming sixth revision.
   _XOPEN_SOURCE_EXTENDED XPG things and X/Open Unix extensions.
   _LARGEFILE_SOURCE	Some more functions for correct standard I/O.
   _LARGEFILE64_SOURCE	Additional functionality from LFS for large files.
   _FILE_OFFSET_BITS=N	Select default filesystem interface.
   _BSD_SOURCE		ISO C, POSIX, and 4.3BSD things.
   _SVID_SOURCE		ISO C, POSIX, and SVID things.
   _GNU_SOURCE		All of the above, plus GNU extensions.
   _REENTRANT		Select additionally reentrant object.
   _THREAD_SAFE		Same as _REENTRANT, often used by other systems.

   The `-ansi' switch to the GNU C compiler defines __STRICT_ANSI__.
   If none of these are defined, the default is to have _SVID_SOURCE,
   _BSD_SOURCE, and _POSIX_SOURCE set to one and _POSIX_C_SOURCE set to
   199506L.  If more than one of these are defined, they accumulate.
   For example __STRICT_ANSI__, _POSIX_SOURCE and _POSIX_C_SOURCE
   together give you ISO C, 1003.1, and 1003.2, but nothing else.

   These are defined by this file and are used by the
   header files to decide what to declare or define:

   __USE_ISOC99		Define ISO C99 things.
   __USE_POSIX		Define IEEE Std 1003.1 things.
   __USE_POSIX2		Define IEEE Std 1003.2 things.
   __USE_POSIX199309	Define IEEE Std 1003.1, and .1b things.
   __USE_POSIX199506	Define IEEE Std 1003.1, .1b, .1c and .1i things.
   __USE_XOPEN		Define XPG things.
   __USE_XOPEN_EXTENDED	Define X/Open Unix things.
   __USE_UNIX98		Define Single Unix V2 things.
   __USE_XOPEN2K        Define XPG6 things.
   __USE_LARGEFILE	Define correct standard I/O things.
   __USE_LARGEFILE64	Define LFS things with separate names.
   __USE_FILE_OFFSET64	Define 64bit interface as default.
   __USE_BSD		Define 4.3BSD things.
   __USE_SVID		Define SVID things.
   __USE_MISC		Define things common to BSD and System V Unix.
   __USE_GNU		Define GNU extensions.
   __USE_REENTRANT	Define reentrant/thread-safe *_r functions.
   __FAVOR_BSD		Favor 4.3BSD things in cases of conflict.

   The macros `__GNU_LIBRARY__', `__GLIBC__', and `__GLIBC_MINOR__' are
   defined by this file unconditionally.  `__GNU_LIBRARY__' is provided
   only for compatibility.  All new code should use the other symbols
   to test for features.

   All macros listed above as possibly being defined by this file are
   explicitly undefined if they are not explicitly defined.
   Feature-test macros that are not defined by the user or compiler
   but are implied by the other feature-test macros defined (or by the
   lack of any definitions) are defined by the file.  */


/* Undefine everything, so we get a clean slate.  */
# 106 "/usr/include/features.h" 3
/* Suppress kernel-name space pollution unless user expressedly asks
   for it.  */




/* Always use ISO C things.  */



/* If _BSD_SOURCE was defined by the user, favor BSD over POSIX.  */







/* If _GNU_SOURCE was defined by the user, turn on all the other features.  */
# 144 "/usr/include/features.h" 3
/* If nothing (other than _GNU_SOURCE) is defined,
   define _BSD_SOURCE and _SVID_SOURCE.  */
# 154 "/usr/include/features.h" 3
/* This is to enable the ISO C99 extension.  Also recognize the old macro
   which was used prior to the standard acceptance.  This macro will
   eventually go away and the features enabled by default once the ISO C99
   standard is widely adopted.  */





/* If none of the ANSI/POSIX macros are defined, use POSIX.1 and POSIX.2
   (and IEEE Std 1003.1b-1993 unless _XOPEN_SOURCE is defined).  */
# 242 "/usr/include/features.h" 3
/* We do support the IEC 559 math functionality, real and complex.  */



/* wchar_t uses ISO 10646-1 (2nd ed., published 2000-09-15) / Unicode 3.0.  */


/* This macro indicates that the installed library is the GNU C Library.
   For historic reasons the value now is 6 and this will stay from now
   on.  The use of this variable is deprecated.  Use __GLIBC__ and
   __GLIBC_MINOR__ now (see below) when you want to test for a specific
   GNU C library version and use the values in <gnu/lib-names.h> to get
   the sonames of the shared libraries.  */



/* Major and minor version number of the GNU C library package.  Use
   these macros to test for features in specific releases.  */



/* Convenience macros to test the versions of glibc and gcc.
   Use them like this:
   #if __GNUC_PREREQ (2,8)
   ... code requiring gcc 2.8 or later ...
   #endif
   Note - they won't work for gcc1 or glibc1, since the _MINOR macros
   were not defined then.  */
# 280 "/usr/include/features.h" 3
/* This is here only because every header file already includes this one.  */


# 1 "/usr/include/sys/cdefs.h" 1 3
/* Copyright (C) 1992,93,94,95,96,97,98,99,2000,2001 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, write to the Free
   Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307 USA.  */




/* We are almost always included from features.h. */




/* The GNU libc does not support any K&R compilers or the traditional mode
   of ISO C compilers anymore.  Check for some of the combinations not
   anymore supported.  */




/* Some user header file might have defined this before.  */





/* GCC can always grok prototypes.  For C++ programs we add throw()
   to help it optimize the function calls.  But this works only with
   gcc 2.8.x and egcs.  */






/* This macro will be used for functions which might take C++ callback
   functions.  */
# 67 "/usr/include/sys/cdefs.h" 3
/* For these things, GCC behaves the ANSI way normally,
   and the non-ANSI way under -traditional.  */




/* This is not a typedef so `const __ptr_t' does the right thing.  */




/* C++ needs to know that types and declarations are C, not C++.  */
# 88 "/usr/include/sys/cdefs.h" 3
/* Support for bounded pointers.  */







/* Support for flexible arrays.  */

/* GCC 2.97 supports C99 flexible array members.  */
# 107 "/usr/include/sys/cdefs.h" 3
/* Some other non-C99 compiler.  Approximate with [1].  */






/* __asm__ ("xyz") is used throughout the headers to rename functions
   at the assembly language level.  This is wrapped by the __REDIRECT
   macro, in order to support compilers that can do this some other
   way.  When compilers don't support asm-names at all, we have to do
   preprocessor tricks instead (which don't have exactly the right
   semantics, but it's the best we can do).

   Example:
   int __REDIRECT(setpgrp, (__pid_t pid, __pid_t pgrp), setpgid); */







/*
#elif __SOME_OTHER_COMPILER__

# define __REDIRECT(name, proto, alias) name proto; \
	_Pragma("let " #name " = " #alias)
*/


/* GCC has various useful declarations that can be made with the
   `__attribute__' syntax.  All of the ways we use this do fine if
   they are omitted for compilers that don't understand it. */




/* At some point during the gcc 2.96 development the `malloc' attribute
   for functions was introduced.  We don't want to use it unconditionally
   (although this would be possible) since it generates warnings.  */






/* At some point during the gcc 2.96 development the `pure' attribute
   for functions was introduced.  We don't want to use it unconditionally
   (although this would be possible) since it generates warnings.  */






/* At some point during the gcc 2.8 development the `format_arg' attribute
   for functions was introduced.  We don't want to use it unconditionally
   (although this would be possible) since it generates warnings.
   If several `format_arg' attributes are given for the same function, in
   gcc-3.0 and older, all but the last one are ignored.  In newer gccs,
   all designated arguments are considered.  */






/* At some point during the gcc 2.97 development the `strfmon' format
   attribute for functions was introduced.  We don't want to use it
   unconditionally (although this would be possible) since it
   generates warnings.  */







/* It is possible to compile containing GCC extensions even if GCC is
   run in pedantic mode if the uses are carefully marked using the
   `__extension__' keyword.  But this is not generally available before
   version 2.8.  */




/* __restrict is known in EGCS 1.2 and above. */




/* ISO C99 also allows to declare arrays as non-overlapping.  The syntax is
     array_name[restrict]
   GCC 3.1 supports this.  */
# 211 "/usr/include/sys/cdefs.h" 3
/* Some other non-C99 compiler.  */
# 284 "/usr/include/features.h" 2 3


/* If we don't have __REDIRECT, prototypes will be missing if
   __USE_FILE_OFFSET64 but not __USE_LARGEFILE[64]. */







/* Decide whether we can define 'extern inline' functions in headers.  */





/* This is here only because every header file already includes this one.  */

/* Get the definitions of all the appropriate `__stub_FUNCTION' symbols.
   <gnu/stubs.h> contains `#define __stub_FUNCTION' when FUNCTION is a stub
   which will always return failure (and set errno to ENOSYS).

   We avoid including <gnu/stubs.h> when compiling the C library itself to
   avoid a dependency loop.  stubs.h depends on every object file.  If
   this #include were done for the library source code, then every object
   file would depend on stubs.h.  */

# 1 "/usr/include/gnu/stubs.h" 1 3
/* This file is automatically generated.
   It defines a symbol `__stub_FUNCTION' for each function
   in the C library which is a stub, meaning it will fail
   every time called, usually setting errno to ENOSYS.  */
# 313 "/usr/include/features.h" 2 3
# 127 "/cdf/home/pcanal/scratch/code/root.merging/include/RConfig.h" 2
# 175 "/cdf/home/pcanal/scratch/code/root.merging/include/RConfig.h"
/*#   define R__B64 */ /* enable when 64 bit machine */






/*#   define R__B64 */ /* enable when 64 bit machine */
# 335 "/cdf/home/pcanal/scratch/code/root.merging/include/RConfig.h"
/*--- memory and object statistics -------------------------------------------*/

/* #define R__NOSTATS */


/*--- cpp --------------------------------------------------------------------*/


    /* symbol concatenation operator */




    /* stringizing */
# 361 "/cdf/home/pcanal/scratch/code/root.merging/include/RConfig.h"
/* produce an indentifier that is almost unique inside a file */







/*---- misc ------------------------------------------------------------------*/
# 25 "/cdf/home/pcanal/scratch/code/root.merging/include/Rtypes.h" 2


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/DllImport.h" 1
/* @(#)root/base:$Name:  $:$Id: DllImport.h,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*
  This include file defines DllImport/DllExport macro
  to build DLLs under Windows OS.

  They are defined as dummy for UNIX's
*/
# 28 "/cdf/home/pcanal/scratch/code/root.merging/include/Rtypes.h" 2


# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/stdio.h" 1
# 31 "/cdf/home/pcanal/scratch/code/root.merging/include/Rtypes.h" 2
# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/Rtypeinfo.h" 1
// @(#)root/base:$Name:  $:$Id: Rtypeinfo.h,v 1.1.2.1 2002/02/25 18:03:29 rdm Exp $
// Author: Philippe Canal   23/2/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
# 21 "/cdf/home/pcanal/scratch/code/root.merging/include/Rtypeinfo.h"
// <typeinfo> includes <exception> which clashes with <math.h>
//#include <typeinfo.h>





# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/typeinfo" 1
namespace std {
# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/typeinfo.h" 1
/*********************************************************************
* typeinfo.h
*
*  Run time type identification
*
* Memo:
*   typeid(typename) , typeid(expression) is implemented as special 
*  function in the cint body src/G__func.c. 
*
*   As an extention, G__typeid(char *name) is defined in src/G__func.c
*  too for more dynamic use of the typeid.
*
*   type_info is extended to support non-polymorphic type objects.
*
*   In src/G__sizeof.c , G__typeid() is implemented. It relies on
*  specific binary layout of type_info object. If order of type_info
*  member declaration is modified, src/G__sizeof.c must be modified
*  too.
*
*********************************************************************/






# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/bool.h" 1
#pragma ifndef G__BOOL_H
#pragma define G__BOOL_H

#pragma ifdef G__OLDIMPLEMENTATION1604
/* This header file may not be needed any more */

//#undef FALSE
//#undef TRUE



// bool as fundamental type
const bool false=0,true=1;



bool bool() { return false; }

// This is not needed due to fix 1584
//#pragma link off class bool;
//#pragma link off function bool;

#pragma endif

#pragma endif
# 28 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/typeinfo.h" 2

/*********************************************************************
* Functions embedded in cint core
* Most of those functions are defined in src/sizeof.c
* 
*********************************************************************/
// type_info typeid(expression);
// type_info typeid(char *typename);
// type_info G__typeid(char *expression);
// long G__get_classinfo(char *item,int tagnum);
// long G__get_variableinfo(char *item,long *handle,long *index,long tagnum);
// long G__get_functioninfo(char *item,long *handle,long &index,long tagnum);


/*********************************************************************
* type_info
*
*  Included in ANSI/ISO resolution proposal 1995 spring
* 
*********************************************************************/
class type_info {
 public:
  virtual ~type_info() { } // type_info is polymorphic
  bool operator==(const type_info&) const;
  bool operator!=(const type_info&) const;
  bool before(const type_info&) const;

  const char* name() const;

 private:
  type_info(const type_info&);
 protected: // original enhancement
  type_info& operator=(const type_info&);

  // implementation dependent representation
 protected:
  long type; // intrinsic types
  long tagnum; // class/struct/union
  long typenum; // typedefs
  long reftype; // pointing level and reference types
  long size; // size of the object

 public: // original enhancement
  type_info() { }
};


bool type_info::operator==(const type_info& a) const
{
  if(reftype == a.reftype && tagnum == a.tagnum && type == a.type)
    return(true);
  else
    return(false);
}

bool type_info::operator!=(const type_info& a) const
{
  if( *this == a ) return(false);
  else return(true);
}

bool type_info::before(const type_info& a) const
{
  if(-1!=tagnum)
    return( tagnum < a.tagnum );
  else if(-1!=a.tagnum)
    return( -1 < a.tagnum );
  else
    return( type < a.type );
}

const char* type_info::name() const
{
  static char namestring[100];
  //printf("%d %d %d %d\n",type,tagnum,typenum,reftype);
  strcpy(namestring,G__type2string(type,tagnum,typenum,reftype));
  return(namestring);
}

type_info::type_info(const type_info& a)
{
  type = a.type;
  tagnum = a.tagnum;
  typenum = a.typenum;
  reftype = a.reftype;
  size = a.size;
}

type_info& type_info::operator=(const type_info& a)
{
  type = a.type;
  tagnum = a.tagnum;
  typenum = a.typenum;
  reftype = a.reftype;
  size = a.size;
  return(*this);
}

/**************************************************************************
* original enhancment
**************************************************************************/
type_info::type_info()
{
  type = 0;
  tagnum = typenum = -1;
  reftype = 0;
}


/**************************************************************************
* Further runtime type checking requirement from Fons Rademaker
**************************************************************************/

/*********************************************************************
* G__class_info
*
*********************************************************************/
class G__class_info : public type_info {
 public:
  G__class_info() { init(); }
  G__class_info(type_info& a) { init(a); }
  G__class_info(char *classname) { init(G__typeid(classname)); }

  void init() {
    typenum = -1;
    reftype = 0;
    tagnum = G__get_classinfo("next",-1);
    size = G__get_classinfo("size",tagnum);
    type = G__get_classinfo("type",tagnum);
  }

  void init(type_info& a) {
    type_info *p=this;
    *p = a;
  }

  G__class_info& operator=(G__class_info& a) {
    type = a.type;
    tagnum = a.tagnum;
    typenum = a.typenum;
    reftype = a.reftype;
    size = a.size;
  }

  G__class_info& operator=(type_info& a) {
    init(a);
  }

  G__class_info* next() {
    tagnum=G__get_classinfo("next",tagnum);
    if(-1!=tagnum) return(this);
    else {
      size = type = 0;
      return((G__class_info*)NULL);
    }
  }

  char *title() {
    return((char*)G__get_classinfo("title",tagnum));
  }

  // char *name() is inherited from type_info

  char *baseclass() {
    return((char*)G__get_classinfo("baseclass",tagnum));
  }


  int isabstract() {
    return((int)G__get_classinfo("isabstract",tagnum));
  }

  // can be implemented
  // int iscompiled();

  int Tagnum() {
    return(tagnum);
  }

};


/*********************************************************************
* G__variable_info
*
*********************************************************************/
class G__variable_info {
 public:
  G__variable_info() { init(); }
  G__variable_info(G__class_info& a) { init(a); }
  G__variable_info(char *classname) { init(G__class_info(classname)); }

  void init() {
    G__get_variableinfo("new",&handle,&index,tagnum=-1);
  }

  void init(G__class_info& a) {
    G__get_variableinfo("new",&handle,&index,tagnum=a.Tagnum());
  }

  G__variable_info* next() {
    if(G__get_variableinfo("next",&handle,&index,tagnum)) return(this);
    else return((G__variable_info*)NULL);
  }

  char *title() {
    return((char*)G__get_variableinfo("title",&handle,&index,tagnum));
  }

  char *name() {
    return((char*)G__get_variableinfo("name",&handle,&index,tagnum));
  }

  char *type() {
    return((char*)G__get_variableinfo("type",&handle,&index,tagnum));
  }

  int offset() {
    return((int)G__get_variableinfo("offset",&handle,&index,tagnum));
  }

  // can be implemented
  // char *access(); // return public,protected,private
  // int isstatic();
  // int iscompiled();

 private:
  long handle; // pointer to variable table
  long index;
  long tagnum; // class/struct identity
};

/*********************************************************************
* G__function_info
*
*********************************************************************/
class G__function_info {
 public:
  G__function_info() { init(); }
  G__function_info(G__class_info& a) { init(a); }
  G__function_info(char *classname) { init(G__class_info(classname)); }

  void init() {
    G__get_functioninfo("new",&handle,&index,tagnum=-1);
  } // initialize for global function

  void init(G__class_info& a) {
    G__get_functioninfo("new",&handle,&index,tagnum=a.Tagnum());
  } // initialize for member function

  G__function_info* next() {
    if(G__get_functioninfo("next",&handle,&index,tagnum)) return(this);
    else return((G__function_info*)NULL);
  }

  char *title() {
    return((char*)G__get_functioninfo("title",&handle,&index,tagnum));
  }

  char *name() {
    return((char*)G__get_functioninfo("name",&handle,&index,tagnum));
  }

  char *type() {
    return((char*)G__get_functioninfo("type",&handle,&index,tagnum));
  }

  char *arglist() {
    return((char*)G__get_functioninfo("arglist",&handle,&index,tagnum));
  }

  // can be implemented
  // char *access(); // return public,protected,private
  // int isstatic();
  // int iscompiled();
  // int isvirtual();
  // int ispurevirtual();

 private:
  long handle; // pointer to variable table
  long index;
  long tagnum; // class/struct identity
};

/*********************************************************************
* G__string_buf
*
*  This struct is used as temporary object for returning title strings.
* Size of buf[] limits maximum length of the title string you can
* describe. You can increase size of it here to increase it.
*
*********************************************************************/
struct G__string_buf {
  char buf[256];
};


/*********************************************************************
* Example code
*
*  Following functions are the examples of how to use the type info
* facilities.
*
*********************************************************************/


void G__list_class(void) {
  G__class_info a;
  do {
    printf("%s:%s =%d '%s'\n",a.name(),a.baseclass(),a.isabstract(),a.title());
  } while(a.next());
}

void G__list_class(char *classname) {
  G__list_memvar(classname);
  G__list_memfunc(classname);
}

void G__list_memvar(char *classname) {
  G__variable_info a=G__variable_info(G__typeid(classname));
  do {
    printf("%s %s; offset=%d '%s'\n",a.type(),a.name(),a.offset(),a.title());
  } while(a.next());
}

void G__list_memfunc(char *classname) {
  G__function_info a=G__function_info(G__typeid(classname));
  do {
    printf("%s %s(%s) '%s'\n",a.type(),a.name(),a.arglist(),a.title());
  } while(a.next());
}
# 3 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/typeinfo" 2
}
# 29 "/cdf/home/pcanal/scratch/code/root.merging/include/Rtypeinfo.h" 2
using std::type_info;
# 32 "/cdf/home/pcanal/scratch/code/root.merging/include/Rtypes.h" 2



//---- forward declared class types --------------------------------------------

class TClass;
class TBuffer;
class TMemberInspector;
class TObject;
class TNamed;

//---- types -------------------------------------------------------------------

typedef char Char_t; //Signed Character 1 byte (char)
typedef unsigned char UChar_t; //Unsigned Character 1 byte (unsigned char)
typedef short Short_t; //Signed Short integer 2 bytes (short)
typedef unsigned short UShort_t; //Unsigned Short integer 2 bytes (unsigned short)

                                    //Signed integer 4 bytes
                                    //Unsigned integer 4 bytes

typedef int Int_t; //Signed integer 4 bytes (int)
typedef unsigned int UInt_t; //Unsigned integer 4 bytes (unsigned int)


                                    //File pointer (int)
                                    //Signed long integer 8 bytes (long)
                                    //Unsigned long integer 8 bytes (unsigned long)

typedef int Seek_t; //File pointer (int)
typedef long Long_t; //Signed long integer 4 bytes (long)
typedef unsigned long ULong_t; //Unsigned long integer 4 bytes (unsigned long)

typedef float Float_t; //Float 4 bytes (float)
typedef double Double_t; //Float 8 bytes (double)
typedef char Text_t; //General string (char)
typedef bool Bool_t; //Boolean (0=false, 1=true) (bool)
typedef unsigned char Byte_t; //Byte (8 bits) (unsigned char)
typedef short Version_t; //Class version identifier (short)
typedef const char Option_t; //Option string (const char)
typedef int Ssiz_t; //String size (int)
typedef float Real_t; //TVector and TMatrix element type (float)

typedef void (*Streamer_t)(TBuffer&, void*, Int_t);
typedef void (*VoidFuncPtr_t)(); //pointer to void function


//---- constants ---------------------------------------------------------------





const Bool_t kTRUE = 1;
const Bool_t kFALSE = 0;

const Int_t kMaxInt = 2147483647;
const Int_t kMaxShort = 32767;
const size_t kBitsPerByte = 8;
const Ssiz_t kNPOS = ~(Ssiz_t)0;


//--- bit manipulation ---------------------------------------------------------







//---- debug global ------------------------------------------------------------

R__EXTERN Int_t gDebug;


//---- ClassDef macros ---------------------------------------------------------

typedef void (*ShowMembersFunc_t)(void *obj, TMemberInspector &R__insp, char *R__parent);
typedef TClass *(*IsAFunc_t)(const void *obj);




   // Read TObject derived classes from a TBuffer. Need to provide
   // custom version for non-TObject derived classes. The const
   // version below is correct for any class.

   // This implementation only works for classes inheriting from
   // TObject.  This enables a clearer error message from the compiler.





template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj);


template <class Tmpl> TBuffer &operator>>(TBuffer &buf, const Tmpl *&obj) {
   return operator>>(buf, (Tmpl *&) obj);
}


// template <class RootClass> Short_t GetClassVersion(RootClass *);

namespace ROOT {
   // NOTE: Cint typeid is not fully functional yet, so these classes can not
   // be made available yet.
   template <class T> TClass *IsA(T *obj) { return gROOT->GetClass(typeid(*obj)); }
   template <class T> TClass *IsA(const T *obj) { return IsA((T*)obj); }

   template <class RootClass> class ClassInfo;

   template <class RootClass> Short_t SetClassVersion();

   extern TClass *CreateClass(const char *cname, Version_t id,
                              const type_info &info, IsAFunc_t isa,
                              ShowMembersFunc_t show,
                              const char *dfil, const char *ifil,
                              Int_t dl, Int_t il);
   extern void AddClass(const char *cname, Version_t id, const type_info &info,
                        VoidFuncPtr_t dict, Int_t pragmabits);
   extern void RemoveClass(const char *cname);
   extern void ResetClassVersion(TClass*, const char*, Short_t);

   extern TNamed *RegisterClassTemplate(const char *name,
                                        const char *file, Int_t line);

   // This function is only implemented in the dictionary file.
   // The parameter is 'only' for overloading resolution.
   template <class T> ClassInfo<T> &GenerateInitInstance(const T*);

   // Because of the template defined here, we have to insure that
   // CINT does not see this file twice, even if it is preprocessed by
   // an external preprocessor.

#pragma define ROOT_Rtypes_In_Cint_Interpreter


#pragma ifndef ROOT_Rtypes_For_Cint
#pragma define ROOT_Rtypes_For_Cint


   class InitBehavior {
   public:
      virtual void Register(const char *cname, Version_t id, const type_info &info,
                            VoidFuncPtr_t dict, Int_t pragmabits) const = 0;
      virtual void Unregister(const char *classname) const = 0;
      virtual TClass *CreateClass(const char *cname, Version_t id,
                                  const type_info &info, IsAFunc_t isa,
                                  ShowMembersFunc_t show,
                                  const char *dfil, const char *ifil,
                                  Int_t dl, Int_t il) const = 0;
   };

   class DefaultInitBehavior : public InitBehavior {
   public:
      virtual void Register(const char *cname, Version_t id, const type_info &info,
                            VoidFuncPtr_t dict, Int_t pragmabits) const {
         ROOT::AddClass(cname, id, info, dict, pragmabits);
      }
      virtual void Unregister(const char *classname) const {
         ROOT::RemoveClass(classname);
      }
      virtual TClass *CreateClass(const char *cname, Version_t id,
                                  const type_info &info, IsAFunc_t isa,
                                  ShowMembersFunc_t show,
                                  const char *dfil, const char *ifil,
                                  Int_t dl, Int_t il) const {
         return ROOT::CreateClass(cname, id, info, isa, show, dfil, ifil, dl, il);
      }
   };

   template <class RootClass > class ClassInfo {
   public:
      typedef void (*ShowMembersFunc_t)(RootClass *obj, TMemberInspector &R__insp,
                                        char *R__parent);
   protected:



     friend ClassInfo<RootClass > &GenerateInitInstance<RootClass >(const RootClass*);


     static const InitBehavior *fgAction;
     static TClass *fgClass;
     static Int_t fgVersion;
     static const char *fgClassName;
     static const char *fgImplFileName;
     static Int_t fgImplFileLine;
     static const char *fgDeclFileName;
     static Int_t fgDeclFileLine;
     static ShowMembersFunc_t fgShowMembers;

     ClassInfo(const char *fullClassname,
               const char *declFilename, Int_t declFileline,
               ShowMembersFunc_t showmembers, Int_t pragmabits) {
        // The basic type global varible are initialized to 0
        Int_t version = 1; // This is the default version number. 
        if (fgVersion !=0) version = fgVersion;
        Init(fullClassname, version,
             declFilename, declFileline,
             showmembers, pragmabits);
     }
     ClassInfo(const char *fullClassname, Int_t version,
               const char *declFilename, Int_t declFileline,
               ShowMembersFunc_t showmembers, Int_t pragmabits) {
        Init(fullClassname, version,
             declFilename, declFileline,
             showmembers, pragmabits);
     }

     void Init(const char *fullClassname, Int_t version,
               const char *declFilename, Int_t declFileline,
               ShowMembersFunc_t showmembers, Int_t pragmabits) {
        GetAction().Register(fullClassname,
                             version,
                             typeid(RootClass),
                             &Dictionary,
                             pragmabits);
        fgShowMembers = showmembers;
        fgVersion = version;
        fgClassName = fullClassname;
        fgDeclFileName = declFilename;
        fgDeclFileLine = declFileline;
     }

  public:
     ~ClassInfo() { GetAction().Unregister(GetClassName()); }

     static const InitBehavior &GetAction() {
        if (!fgAction) {
           RootClass *ptr = 0;
           fgAction = DefineBehavior(ptr, ptr);
        }
        return *fgAction;
     }

     static void Dictionary() { GetClass(); }

     static TClass *GetClass() {
        if (!fgClass) {
           GenerateInitInstance((const RootClass*)0x0);
           fgClass = GetAction().CreateClass(GetClassName(),
                                             GetVersion(),
                                             typeid(RootClass),
                                             &IsA,
                                             &ShowMembers,
                                             GetDeclFileName(),
                                             GetImplFileName(),
                                             GetDeclFileLine(),
                                             GetImplFileLine());
        }
        return fgClass;
     }

     static const char *GetClassName() {
        return fgClassName;
     }

     static ShowMembersFunc_t GetShowMembers() {
        return fgShowMembers;
     }

     static Short_t SetVersion(Short_t version) {
        ROOT::ResetClassVersion(fgClass, GetClassName(),version);
        fgVersion = version;
        return version;
     }

     static void SetFromTemplate() {
        TNamed *info = ROOT::RegisterClassTemplate(GetClassName(), 0, 0);
        if (info) SetImplFile(info->GetTitle(), info->GetUniqueID());
     }

     static int SetImplFile(const char *file, Int_t line) {
        fgImplFileName = file;
        fgImplFileLine = line;
        return 0;
     }

     static const char *GetDeclFileName() {
        return fgDeclFileName;
     }

     static Int_t GetDeclFileLine() {
        return fgDeclFileLine;
     }

     static const char *GetImplFileName() {
        if (!fgImplFileName) SetFromTemplate();
        return fgImplFileName;
     }

     static Int_t GetImplFileLine() {
        if (!fgImplFileLine) SetFromTemplate();
        return fgImplFileLine;
     }

     static Int_t GetVersion() {
        return fgVersion;
     }

     static void ShowMembers(RootClass *obj, TMemberInspector &R__insp,
                             char *R__parent) {
        if (fgShowMembers) fgShowMembers(obj, R__insp, R__parent);
        // for now other part of the system seem to warn about this,
        // so we can just do as if the class was 'empty'
        // else
        //Error("R__tInit","ShowMembers not initialized for %s",GetClassName());
     }

     static TClass* IsA(const void *obj) {
        return ROOT::IsA( (RootClass*)obj );
     }

  protected:
     static void ShowMembers(void *obj, TMemberInspector &R__insp,
                             char *R__parent) {
        if (fgShowMembers) fgShowMembers((RootClass*)obj,R__insp,R__parent);
        // for now other part of the system seem to warn about this,
        // so we can just do as if the class was 'empty'
        // else
        //Error("R__tInit","ShowMembers not initialized for %s",GetClassName());
     }
  };


#pragma endif


} // End of namespace ROOT
# 411 "/cdf/home/pcanal/scratch/code/root.merging/include/Rtypes.h"
//---- ClassDefT macros for templates with one template argument ---------------
// ClassDefT  corresponds to ClassDef
// ClassDefT2 goes in the same header as ClassDefT but must be
//            outside the class scope
// ClassImpT  corresponds to ClassImp


// For now we keep ClassDefT a simple macro to avoid cint parser related issues.
# 468 "/cdf/home/pcanal/scratch/code/root.merging/include/Rtypes.h"
//---- ClassDefT macros for templates with two template arguments --------------
// ClassDef2T2 goes in the same header as ClassDefT but must be
//             outside the class scope
// ClassImp2T  corresponds to ClassImpT





//---- ClassDefT macros for templates with three template arguments ------------
// ClassDef3T2 goes in the same header as ClassDefT but must be
//             outside the class scope
// ClassImp3T  corresponds to ClassImpT





//---- Macro to set the class version of non instrumented class an implementation file -----
# 32 "/cdf/home/pcanal/scratch/code/root.merging/include/TObject.h" 2


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/Varargs.h" 1
/* @(#)root/base:$Name:  $:$Id: Varargs.h,v 1.2 2001/01/18 11:26:50 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





/* typedef char *va_list; */
# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/stdarg.h" 1
/****************************************************************
* stdarg.h
*****************************************************************/



struct va_list {
  void* libp;
  int ip;
} ;


/* not needed anymore */
# 17 "/cdf/home/pcanal/scratch/code/root.merging/include/Varargs.h" 2
# 35 "/cdf/home/pcanal/scratch/code/root.merging/include/TObject.h" 2


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TStorage.h" 1
// @(#)root/base:$Name:  $:$Id: TStorage.h,v 1.4 2001/11/16 02:36:13 rdm Exp $
// Author: Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStorage                                                             //
//                                                                      //
// Storage manager.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////





typedef void (*FreeHookFun_t)(void*, void *addr, size_t);
typedef void *(*ReAllocFun_t)(void*, size_t);
typedef void *(*ReAllocCFun_t)(void*, size_t, size_t);


class TStorage {

private:
   static ULong_t fgHeapBegin; // begin address of heap
   static ULong_t fgHeapEnd; // end address of heap
   static size_t fgMaxBlockSize; // largest block allocated
   static FreeHookFun_t fgFreeHook; // function called on free
   static void *fgFreeHookData; // data used by this function
   static ReAllocFun_t fgReAllocHook; // custom ReAlloc
   static ReAllocCFun_t fgReAllocCHook; // custom ReAlloc with length check
   static Bool_t fgHasCustomNewDelete; // true if using ROOT's new/delete

public:
   static ULong_t GetHeapBegin();
   static ULong_t GetHeapEnd();
   static FreeHookFun_t GetFreeHook();
   static void *GetFreeHookData();
   static size_t GetMaxBlockSize();
   static void *ReAlloc(void *vp, size_t size);
   static void *ReAlloc(void *vp, size_t size, size_t oldsize);
   static char *ReAllocChar(char *vp, size_t size, size_t oldsize);
   static Int_t *ReAllocInt(Int_t *vp, size_t size, size_t oldsize);
   static void *ObjectAlloc(size_t size);
   static void *ObjectAlloc(size_t size, void *vp);
   static void ObjectDealloc(void *vp);
   static void ObjectDealloc(void *vp, void *ptr);

   static void EnterStat(size_t size, void *p);
   static void RemoveStat(void *p);
   static void PrintStatistics();
   static void SetMaxBlockSize(size_t size);
   static void SetFreeHook(FreeHookFun_t func, void *data);
   static void SetReAllocHooks(ReAllocFun_t func1, ReAllocCFun_t func2);
   static void SetCustomNewDelete();
   static void EnableStatistics(int size= -1, int ix= -1);

   static Bool_t HasCustomNewDelete();

   // only valid after call to a TStorage allocating method
   static void AddToHeap(ULong_t begin, ULong_t end);
   static Bool_t IsOnHeap(void *p);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 0; } static void Dictionary(); virtual TClass *IsA() const { return TStorage::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TStorage::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TStorage.h"; } static int DeclFileLine() { return 75; } static const char *ImplFileName(); static int ImplFileLine(); //Storage manager class
};


inline void TStorage::AddToHeap(ULong_t begin, ULong_t end)
   { if (begin < fgHeapBegin) fgHeapBegin = begin;
     if (end > fgHeapEnd) fgHeapEnd = end; }

inline Bool_t TStorage::IsOnHeap(void *p)
   { return (ULong_t)p >= fgHeapBegin && (ULong_t)p < fgHeapEnd; }

inline size_t TStorage::GetMaxBlockSize() { return fgMaxBlockSize; }

inline void TStorage::SetMaxBlockSize(size_t size) { fgMaxBlockSize = size; }

inline FreeHookFun_t TStorage::GetFreeHook() { return fgFreeHook; }
# 38 "/cdf/home/pcanal/scratch/code/root.merging/include/TObject.h" 2


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/Riosfwd.h" 1
// @(#)root/base:$Name:  $:$Id: Riosfwd.h,v 1.1 2002/01/24 11:39:26 rdm Exp $
// Author: Fons Rademakers   23/1/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
# 23 "/cdf/home/pcanal/scratch/code/root.merging/include/Riosfwd.h"
# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iosfwd" 1
namespace std {
# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iosfwd.h" 1


# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h" 1
# 4 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iosfwd.h" 2
# 3 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iosfwd" 2
}
# 24 "/cdf/home/pcanal/scratch/code/root.merging/include/Riosfwd.h" 2

using namespace std;
# 41 "/cdf/home/pcanal/scratch/code/root.merging/include/TObject.h" 2






class TList;
class TBrowser;
class TBuffer;
class TObjArray;
class TMethod;
class TTimer;


//----- Global bits (can be set for any object and should not be reused).
//----- Bits 0 - 13 are reserved as global bits. Bits 14 - 23 can be used
//----- in different class hierarchies (make sure there is no overlap in
//----- any given hierarchy).
enum EObjBits {
   kCanDelete = (1 << (0)), // if object in a list can be deleted
   kMustCleanup = (1 << (3)), // if object destructor must call RecursiveRemove()
   kObjInCanvas = (1 << (3)), // for backward compatibility only, use kMustCleanup
   kIsReferenced = (1 << (4)), // if object is referenced by a TRef or TRefArray
   kCannotPick = (1 << (6)), // if object in a pad cannot be picked
   kNoContextMenu = (1 << (8)), // if object does not want context menu
   kInvalidObject = (1 << (13)) // if object ctor succeeded but object should not be used
};


class TObject {

private:
   UInt_t fUniqueID; //object unique identifier
   UInt_t fBits; //bit field status word

   static Long_t fgDtorOnly; //object for which to call dtor only (i.e. no delete)
   static Bool_t fgObjectStat; //if true keep track of objects in TObjectTable

protected:
   void MakeZombie() { fBits |= kZombie; }
   void DoError(int level, const char *location, const char *fmt, va_list va) const;

public:
   //----- Private bits, clients can only test but not change them
   enum {
      kIsOnHeap = 0x01000000, // object is on heap
      kNotDeleted = 0x02000000, // object has not been deleted
      kZombie = 0x04000000, // object ctor failed
      kBitMask = 0x00ffffff
   };

   //----- Write() options
   enum {
      kSingleKey = (1 << (0)), // write collection with single key
      kOverwrite = (1 << (1)) // overwrite existing object with same name
   };

   TObject();
   TObject(const TObject &object);
   TObject &operator=(const TObject &rhs);
   virtual ~TObject();

   void AppendPad(Option_t *option="");
   virtual void Browse(TBrowser *b);
   virtual const char *ClassName() const;
   virtual void Clear(Option_t * /*option*/ ="") { }
   virtual TObject *Clone(const char *newname="") const;
   virtual Int_t Compare(const TObject *obj) const;
   virtual void Copy(TObject &object);
   virtual void Delete(Option_t *option=""); // *MENU*
   virtual Int_t DistancetoPrimitive(Int_t px, Int_t py);
   virtual void Draw(Option_t *option="");
   virtual void DrawClass() const; // *MENU*
   virtual TObject *DrawClone(Option_t *option="") const; // *MENU*
   virtual void Dump() const; // *MENU*
   virtual void Execute(const char *method, const char *params, Int_t *error=0);
   virtual void Execute(TMethod *method, TObjArray *params, Int_t *error=0);
   virtual void ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual TObject *FindObject(const char *name) const;
   virtual TObject *FindObject(const TObject *obj) const;
   virtual Option_t *GetDrawOption() const;
   virtual UInt_t GetUniqueID() const;
   virtual const char *GetName() const;
   virtual const char *GetIconName() const;
   virtual Option_t *GetOption() const { return ""; }
   virtual char *GetObjectInfo(Int_t px, Int_t py) const;
   virtual const char *GetTitle() const;
   virtual Bool_t HandleTimer(TTimer *timer);
   virtual ULong_t Hash() const;
   virtual Bool_t InheritsFrom(const char *classname) const;
   virtual Bool_t InheritsFrom(const TClass *cl) const;
   virtual void Inspect() const; // *MENU*
   virtual Bool_t IsFolder() const;
   virtual Bool_t IsEqual(const TObject *obj) const;
   virtual Bool_t IsSortable() const { return kFALSE; }
           Bool_t IsOnHeap() const { return TestBit(kIsOnHeap); }
           Bool_t IsZombie() const { return TestBit(kZombie); }
   virtual Bool_t Notify();
   virtual void ls(Option_t *option="") const;
   virtual void Paint(Option_t *option="");
   virtual void Pop();
   virtual void Print(Option_t *option="") const;
   virtual Int_t Read(const char *name);
   virtual void RecursiveRemove(TObject *obj);
   virtual void SavePrimitive(ofstream &out, Option_t *option);
   virtual void SetDrawOption(Option_t *option=""); // *MENU*
   virtual void SetUniqueID(UInt_t uid);
   virtual void UseCurrentStyle();
   virtual Int_t Write(const char *name=0, Int_t option=0, Int_t bufsize=0);

   //----- operators
   void *operator new(size_t sz) { return TStorage::ObjectAlloc(sz); }
   void *operator new(size_t sz, void *vp) { return TStorage::ObjectAlloc(sz, vp); }
   void operator delete(void *ptr);

   void operator delete(void *ptr, void *vp);


   //----- bit manipulation
   void SetBit(UInt_t f, Bool_t set);
   void SetBit(UInt_t f) { fBits |= f & kBitMask; }
   void ResetBit(UInt_t f) { fBits &= ~(f & kBitMask); }
   Bool_t TestBit(UInt_t f) const { return (Bool_t) ((fBits & f) != 0); }
   Int_t TestBits(UInt_t f) const { return (Int_t) (fBits & f); }
   void InvertBit(UInt_t f) { fBits ^= f & kBitMask; }

   //---- error handling
   void Info(const char *method, const char *msgfmt, ...) const;
   void Warning(const char *method, const char *msgfmt, ...) const;
   void Error(const char *method, const char *msgfmt, ...) const;
   void SysError(const char *method, const char *msgfmt, ...) const;
   void Fatal(const char *method, const char *msgfmt, ...) const;

   void AbstractMethod(const char *method) const;
   void MayNotUse(const char *method) const;

   //---- static functions
   static Long_t GetDtorOnly();
   static void SetDtorOnly(void *obj);
   static Bool_t GetObjectStat();
   static void SetObjectStat(Bool_t stat);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 1; } static void Dictionary(); virtual TClass *IsA() const { return TObject::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TObject::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TObject.h"; } static int DeclFileLine() { return 183; } static const char *ImplFileName(); static int ImplFileLine(); //Basic ROOT object
};


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TBuffer.h" 1
// @(#)root/base:$Name:  $:$Id: TBuffer.h,v 1.13 2002/03/18 18:28:03 brun Exp $
// Author: Fons Rademakers   04/05/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBuffer                                                              //
//                                                                      //
// Buffer base class used for serializing objects.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////





# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h" 1
/* @(#)root/base:$Name:  $:$Id: Bytes.h,v 1.8 2002/02/26 11:11:19 brun Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Bytes                                                                //
//                                                                      //
// A set of inline byte handling routines.                              //
//                                                                      //
// The set of tobuf() and frombuf() routines take care of packing a     //
// basic type value into a buffer in network byte order (i.e. they      //
// perform byte swapping when needed). The buffer does not have to      //
// start on a machine (long) word boundary.                             //
//                                                                      //
// For __GNUC__ on linux on i486 processors and up                      //
// use the `bswap' opcode provided by the GNU C Library.                //
//                                                                      //
// The set of host2net() and net2host() routines convert a basic type   //
// value from host to network byte order and vice versa. On BIG ENDIAN  //
// machines this is a no op.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
# 48 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
//______________________________________________________________________________
inline void tobuf(char *&buf, Bool_t x)
{
   UChar_t x1 = x;
   *buf++ = x1;
}

inline void tobuf(char *&buf, UChar_t x)
{
   *buf++ = x;
}

inline void tobuf(char *&buf, UShort_t x)
{


   *((UShort_t *)buf) = Rbswap_16(x);
# 73 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
   buf += sizeof(UShort_t);
}

inline void tobuf(char *&buf, UInt_t x)
{


   *((UInt_t *)buf) = Rbswap_32(x);
# 91 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
   buf += sizeof(UInt_t);
}

inline void tobuf(char *&buf, ULong_t x)
{

   char *sw = (char *)&x;
   if (sizeof(ULong_t) == 8) {
      buf[0] = sw[7];
      buf[1] = sw[6];
      buf[2] = sw[5];
      buf[3] = sw[4];
      buf[4] = sw[3];
      buf[5] = sw[2];
      buf[6] = sw[1];
      buf[7] = sw[0];
   } else {
      buf[0] = 0;
      buf[1] = 0;
      buf[2] = 0;
      buf[3] = 0;
      buf[4] = sw[3];
      buf[5] = sw[2];
      buf[6] = sw[1];
      buf[7] = sw[0];
   }
# 128 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
   buf += 8;
}

inline void tobuf(char *&buf, Float_t x)
{


   *((UInt_t *)buf) = Rbswap_32(*((UInt_t *)&x));

   // Use an union to prevent over-zealous optimization by KCC
   // related to aliasing double.
   // + Use a volatile here to work around error in KCC optimizer
# 159 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
   buf += sizeof(Float_t);
}

inline void tobuf(char *&buf, Double_t x)
{




   // Use an union to prevent over-zealous optimization by KCC
   // related to aliasing double.
   // + Use a volatile here to work around error in KCC optimizer
# 185 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
   char *sw = (char *)&x;
   buf[0] = sw[7];
   buf[1] = sw[6];
   buf[2] = sw[5];
   buf[3] = sw[4];
   buf[4] = sw[3];
   buf[5] = sw[2];
   buf[6] = sw[1];
   buf[7] = sw[0];




   buf += sizeof(Double_t);
}

inline void frombuf(char *&buf, Bool_t *x)
{
   UChar_t x1;
   x1 = *buf++;
   *x = x1;
}

inline void frombuf(char *&buf, UChar_t *x)
{
   *x = *buf++;
}

inline void frombuf(char *&buf, UShort_t *x)
{


   *x = Rbswap_16(*((UShort_t *)buf));
# 226 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
   buf += sizeof(UShort_t);
}

inline void frombuf(char *&buf, UInt_t *x)
{


   *x = Rbswap_32(*((UInt_t *)buf));
# 244 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
   buf += sizeof(UInt_t);
}

inline void frombuf(char *&buf, ULong_t *x)
{

   char *sw = (char *)x;
   if (sizeof(ULong_t) == 8) {
      sw[0] = buf[7];
      sw[1] = buf[6];
      sw[2] = buf[5];
      sw[3] = buf[4];
      sw[4] = buf[3];
      sw[5] = buf[2];
      sw[6] = buf[1];
      sw[7] = buf[0];
   } else {
      sw[0] = buf[7];
      sw[1] = buf[6];
      sw[2] = buf[5];
      sw[3] = buf[4];
   }







   buf += 8;
}

inline void frombuf(char *&buf, Float_t *x)
{


   *((UInt_t*)x) = Rbswap_32(*((UInt_t *)buf));

   // Use an union to prevent over-zealous optimization by KCC
   // related to aliasing double.
   // + Use a volatile here to work around error in KCC optimizer
# 304 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
   buf += sizeof(Float_t);
}

inline void frombuf(char *&buf, Double_t *x)
{




   // Use an union to prevent over-zealous optimization by KCC
   // related to aliasing double.
   // + Use a volatile here to work around error in KCC optimizer
# 330 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
   char *sw = (char *)x;
   sw[0] = buf[7];
   sw[1] = buf[6];
   sw[2] = buf[5];
   sw[3] = buf[4];
   sw[4] = buf[3];
   sw[5] = buf[2];
   sw[6] = buf[1];
   sw[7] = buf[0];




   buf += sizeof(Double_t);
}

inline void tobuf(char *&buf, Char_t x) { tobuf(buf, (UChar_t) x); }
inline void tobuf(char *&buf, Short_t x) { tobuf(buf, (UShort_t) x); }
inline void tobuf(char *&buf, Int_t x) { tobuf(buf, (UInt_t) x); }
inline void tobuf(char *&buf, Long_t x) { tobuf(buf, (ULong_t) x); }

inline void frombuf(char *&buf, Char_t *x) { frombuf(buf, (UChar_t *) x); }
inline void frombuf(char *&buf, Short_t *x) { frombuf(buf, (UShort_t *) x); }
inline void frombuf(char *&buf, Int_t *x) { frombuf(buf, (UInt_t *) x); }
inline void frombuf(char *&buf, Long_t *x) { frombuf(buf, (ULong_t *) x); }


//______________________________________________________________________________

inline UShort_t host2net(UShort_t x)
{

   return Rbswap_16(x);



}

inline UInt_t host2net(UInt_t x)
{

   return Rbswap_32(x);




}

inline ULong_t host2net(ULong_t x)
{
# 399 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
   return (ULong_t)host2net((UInt_t) x);

}

inline Float_t host2net(Float_t xx)
{

   UInt_t t = Rbswap_32(*((UInt_t *)&xx));
   return *(Float_t *)&t;






}

inline Double_t host2net(Double_t x)
{




   char sw[sizeof(Double_t)];
   *(Double_t *)sw = x;

   char *sb = (char *)&x;
   sb[0] = sw[7];
   sb[1] = sw[6];
   sb[2] = sw[5];
   sb[3] = sw[4];
   sb[4] = sw[3];
   sb[5] = sw[2];
   sb[6] = sw[1];
   sb[7] = sw[0];
   return x;

}
# 445 "/cdf/home/pcanal/scratch/code/root.merging/include/Bytes.h"
inline Short_t host2net(Short_t x) { return host2net((UShort_t)x); }
inline Int_t host2net(Int_t x) { return host2net((UInt_t)x); }
inline Long_t host2net(Long_t x) { return host2net((ULong_t)x); }

inline UShort_t net2host(UShort_t x) { return host2net(x); }
inline Short_t net2host(Short_t x) { return host2net(x); }
inline UInt_t net2host(UInt_t x) { return host2net(x); }
inline Int_t net2host(Int_t x) { return host2net(x); }
inline ULong_t net2host(ULong_t x) { return host2net(x); }
inline Long_t net2host(Long_t x) { return host2net(x); }
inline Float_t net2host(Float_t x) { return host2net(x); }
inline Double_t net2host(Double_t x) { return host2net(x); }
# 29 "/cdf/home/pcanal/scratch/code/root.merging/include/TBuffer.h" 2


class TClass;
class TExMap;

class TBuffer : public TObject {

protected:
   Bool_t fMode; //Read or write mode
   Int_t fVersion; //Buffer format version
   Int_t fBufSize; //Size of buffer
   char *fBuffer; //Buffer used to store objects
   char *fBufCur; //Current position in buffer
   char *fBufMax; //End of buffer
   Int_t fMapCount; //Number of objects or classes in map
   Int_t fMapSize; //Default size of map
   Int_t fDisplacement; //Value to be added to the map offsets
   TExMap *fMap; //Map containing object,id pairs for reading/ writing
   TObject *fParent; //Pointer to the buffer parent (file) where buffer is read/written

   enum { kIsOwner = (1 << (14)) }; //If set TBuffer owns fBuffer

   static Int_t fgMapSize; //Default map size for all TBuffer objects

   // Default ctor
   TBuffer() : fMode(0), fBuffer(0) { fMap = 0; fParent = 0;}

   // TBuffer objects cannot be copied or assigned
   TBuffer(const TBuffer &); // not implemented
   void operator=(const TBuffer &); // not implemented

   void CheckCount(UInt_t offset);
   UInt_t CheckObject(UInt_t offset, const TClass *cl, Bool_t readClass = kFALSE);

   void Expand(Int_t newsize); //Expand buffer to newsize

   Int_t Read(const char *name) { return TObject::Read(name); }
   Int_t Write(const char *name, Int_t opt, Int_t bufs)
                                { return TObject::Write(name, opt, bufs); }

public:
   enum EMode { kRead = 0, kWrite = 1 };
   enum { kInitialSize = 1024, kMinimalSize = 128 };
   enum { kMapSize = 503 };

   TBuffer(EMode mode);
   TBuffer(EMode mode, Int_t bufsiz);
   TBuffer(EMode mode, Int_t bufsiz, void *buf, Bool_t adopt = kTRUE);
   virtual ~TBuffer();

   void MapObject(const TObject *obj, UInt_t offset = 1);
   void MapObject(const void *obj, UInt_t offset = 1);
   virtual void Reset() { SetBufferOffset(); ResetMap(); }
   void InitMap();
   void ResetMap();
   void SetReadMode();
   void SetReadParam(Int_t mapsize);
   void SetWriteMode();
   void SetWriteParam(Int_t mapsize);
   void SetBuffer(void *buf, UInt_t bufsiz = 0, Bool_t adopt = kTRUE);
   void SetBufferOffset(Int_t offset = 0) { fBufCur = fBuffer+offset; }
   void SetParent(TObject *parent);
   TObject *GetParent() const;
   char *Buffer() const { return fBuffer; }
   Int_t BufferSize() const { return fBufSize; }
   void DetachBuffer() { fBuffer = 0; }
   Int_t Length() const { return (Int_t)(fBufCur - fBuffer); }

   Int_t CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss);
   void SetByteCount(UInt_t cntpos, Bool_t packInVersion = kFALSE);

   Bool_t IsReading() const { return (fMode & kWrite) == 0; }
   Bool_t IsWriting() const { return (fMode & kWrite) != 0; }

   Int_t ReadBuf(void *buf, Int_t max);
   void WriteBuf(const void *buf, Int_t max);

   char *ReadString(char *s, Int_t max);
   void WriteString(const char *s);

   Version_t ReadVersion(UInt_t *start = 0, UInt_t *bcnt = 0);
   UInt_t WriteVersion(const TClass *cl, Bool_t useBcnt = kFALSE);

   virtual TClass *ReadClass(const TClass *cl = 0, UInt_t *objTag = 0);
   virtual void WriteClass(const TClass *cl);

   virtual TObject *ReadObject(const TClass *cl);
   virtual void WriteObject(const TObject *obj);

   //TObject *ReadObject(const TClass *cl);
   void WriteObject(const void *obj, TClass *actualClass);

   void SetBufferDisplacement(Int_t skipped)
            { fDisplacement = (Int_t)(Length() - skipped); }
   void SetBufferDisplacement() { fDisplacement = 0; }
   Int_t GetBufferDisplacement() const { return fDisplacement; }

   Int_t ReadArray(Bool_t *&b);
   Int_t ReadArray(Char_t *&c);
   Int_t ReadArray(UChar_t *&c);
   Int_t ReadArray(Short_t *&h);
   Int_t ReadArray(UShort_t *&h);
   Int_t ReadArray(Int_t *&i);
   Int_t ReadArray(UInt_t *&i);
   Int_t ReadArray(Long_t *&l);
   Int_t ReadArray(ULong_t *&l);
   Int_t ReadArray(Float_t *&f);
   Int_t ReadArray(Double_t *&d);

   Int_t ReadStaticArray(Bool_t *b);
   Int_t ReadStaticArray(Char_t *c);
   Int_t ReadStaticArray(UChar_t *c);
   Int_t ReadStaticArray(Short_t *h);
   Int_t ReadStaticArray(UShort_t *h);
   Int_t ReadStaticArray(Int_t *i);
   Int_t ReadStaticArray(UInt_t *i);
   Int_t ReadStaticArray(Long_t *l);
   Int_t ReadStaticArray(ULong_t *l);
   Int_t ReadStaticArray(Float_t *f);
   Int_t ReadStaticArray(Double_t *d);

   void WriteArray(const Bool_t *b, Int_t n);
   void WriteArray(const Char_t *c, Int_t n);
   void WriteArray(const UChar_t *c, Int_t n);
   void WriteArray(const Short_t *h, Int_t n);
   void WriteArray(const UShort_t *h, Int_t n);
   void WriteArray(const Int_t *i, Int_t n);
   void WriteArray(const UInt_t *i, Int_t n);
   void WriteArray(const Long_t *l, Int_t n);
   void WriteArray(const ULong_t *l, Int_t n);
   void WriteArray(const Float_t *f, Int_t n);
   void WriteArray(const Double_t *d, Int_t n);

   void ReadFastArray(Bool_t *b, Int_t n);
   void ReadFastArray(Char_t *c, Int_t n);
   void ReadFastArray(UChar_t *c, Int_t n);
   void ReadFastArray(Short_t *h, Int_t n);
   void ReadFastArray(UShort_t *h, Int_t n);
   void ReadFastArray(Int_t *i, Int_t n);
   void ReadFastArray(UInt_t *i, Int_t n);
   void ReadFastArray(Long_t *l, Int_t n);
   void ReadFastArray(ULong_t *l, Int_t n);
   void ReadFastArray(Float_t *f, Int_t n);
   void ReadFastArray(Double_t *d, Int_t n);

   void WriteFastArray(const Bool_t *b, Int_t n);
   void WriteFastArray(const Char_t *c, Int_t n);
   void WriteFastArray(const UChar_t *c, Int_t n);
   void WriteFastArray(const Short_t *h, Int_t n);
   void WriteFastArray(const UShort_t *h, Int_t n);
   void WriteFastArray(const Int_t *i, Int_t n);
   void WriteFastArray(const UInt_t *i, Int_t n);
   void WriteFastArray(const Long_t *l, Int_t n);
   void WriteFastArray(const ULong_t *l, Int_t n);
   void WriteFastArray(const Float_t *f, Int_t n);
   void WriteFastArray(const Double_t *d, Int_t n);

   TBuffer &operator>>(Bool_t &b);
   TBuffer &operator>>(Char_t &c);
   TBuffer &operator>>(UChar_t &c);
   TBuffer &operator>>(Short_t &h);
   TBuffer &operator>>(UShort_t &h);
   TBuffer &operator>>(Int_t &i);
   TBuffer &operator>>(UInt_t &i);
   TBuffer &operator>>(Long_t &l);
   TBuffer &operator>>(ULong_t &l);
   TBuffer &operator>>(Float_t &f);
   TBuffer &operator>>(Double_t &d);
   TBuffer &operator>>(Char_t *c);

   TBuffer &operator<<(Bool_t b);
   TBuffer &operator<<(Char_t c);
   TBuffer &operator<<(UChar_t c);
   TBuffer &operator<<(Short_t h);
   TBuffer &operator<<(UShort_t h);
   TBuffer &operator<<(Int_t i);
   TBuffer &operator<<(UInt_t i);
   TBuffer &operator<<(Long_t l);
   TBuffer &operator<<(ULong_t l);
   TBuffer &operator<<(Float_t f);
   TBuffer &operator<<(Double_t d);
   TBuffer &operator<<(const Char_t *c);

   //friend TBuffer  &operator>>(TBuffer &b, TObject *&obj);
   //friend TBuffer  &operator>>(TBuffer &b, const TObject *&obj);
   friend TBuffer &operator<<(TBuffer &b, const TObject *obj);

   static void SetGlobalReadParam(Int_t mapsize);
   static void SetGlobalWriteParam(Int_t mapsize);
   static Int_t GetGlobalReadParam();
   static Int_t GetGlobalWriteParam();

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 0; } static void Dictionary(); virtual TClass *IsA() const { return TBuffer::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TBuffer::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TBuffer.h"; } static int DeclFileLine() { return 221; } static const char *ImplFileName(); static int ImplFileLine(); //Buffer base class used for serializing objects
};

//---------------------- TBuffer inlines ---------------------------------------

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Bool_t b)
{
   if (fBufCur + sizeof(UChar_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, b);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Char_t c)
{
   if (fBufCur + sizeof(Char_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, c);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Short_t h)
{
   if (fBufCur + sizeof(Short_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, h);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Int_t i)
{
   if (fBufCur + sizeof(Int_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, i);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Long_t l)
{
   if (fBufCur + sizeof(Long_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, l);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Float_t f)
{
   if (fBufCur + sizeof(Float_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, f);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Double_t d)
{
   if (fBufCur + sizeof(Double_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, d);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(const Char_t *c)
{
   WriteString(c);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Bool_t &b)
{
   frombuf(fBufCur, &b);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Char_t &c)
{
   frombuf(fBufCur, &c);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Short_t &h)
{
   frombuf(fBufCur, &h);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Int_t &i)
{
   frombuf(fBufCur, &i);
   return *this;
}

//______________________________________________________________________________
//inline TBuffer &TBuffer::operator>>(Long_t &l)
//{
//   frombuf(fBufCur, &l);
//   return *this;
//}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Float_t &f)
{
   frombuf(fBufCur, &f);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Double_t &d)
{
   frombuf(fBufCur, &d);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Char_t *c)
{
   ReadString(c, -1);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(UChar_t c)
   { return TBuffer::operator<<((Char_t)c); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(UShort_t h)
   { return TBuffer::operator<<((Short_t)h); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(UInt_t i)
   { return TBuffer::operator<<((Int_t)i); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(ULong_t l)
   { return TBuffer::operator<<((Long_t)l); }

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(UChar_t &c)
   { return TBuffer::operator>>((Char_t&)c); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(UShort_t &h)
   { return TBuffer::operator>>((Short_t&)h); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(UInt_t &i)
   { return TBuffer::operator>>((Int_t&)i); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(ULong_t &l)
   { return TBuffer::operator>>((Long_t&)l); }

//______________________________________________________________________________
inline TBuffer &operator<<(TBuffer &buf, const TObject *obj)
   { buf.WriteObject(obj); return buf; }
//______________________________________________________________________________
//inline TBuffer &operator>>(TBuffer &buf, TObject *&obj)
//   { obj = buf.ReadObject(0); return buf; }
//______________________________________________________________________________
//inline TBuffer &operator>>(TBuffer &buf, const TObject *&obj)
//   { obj = buf.ReadObject(0); return buf; }

//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(UChar_t *&c)
   { return TBuffer::ReadArray((Char_t *&)c); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(UShort_t *&h)
   { return TBuffer::ReadArray((Short_t *&)h); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(UInt_t *&i)
   { return TBuffer::ReadArray((Int_t *&)i); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(ULong_t *&l)
   { return TBuffer::ReadArray((Long_t *&)l); }

//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(UChar_t *c)
   { return TBuffer::ReadStaticArray((Char_t *)c); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(UShort_t *h)
   { return TBuffer::ReadStaticArray((Short_t *)h); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(UInt_t *i)
   { return TBuffer::ReadStaticArray((Int_t *)i); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(ULong_t *l)
   { return TBuffer::ReadStaticArray((Long_t *)l); }

//______________________________________________________________________________
inline void TBuffer::ReadFastArray(UChar_t *c, Int_t n)
   { TBuffer::ReadFastArray((Char_t *)c, n); }
//______________________________________________________________________________
inline void TBuffer::ReadFastArray(UShort_t *h, Int_t n)
   { TBuffer::ReadFastArray((Short_t *)h, n); }
//______________________________________________________________________________
inline void TBuffer::ReadFastArray(UInt_t *i, Int_t n)
   { TBuffer::ReadFastArray((Int_t *)i, n); }
//______________________________________________________________________________
inline void TBuffer::ReadFastArray(ULong_t *l, Int_t n)
   { TBuffer::ReadFastArray((Long_t *)l, n); }

//______________________________________________________________________________
inline void TBuffer::WriteArray(const UChar_t *c, Int_t n)
   { TBuffer::WriteArray((const Char_t *)c, n); }
//______________________________________________________________________________
inline void TBuffer::WriteArray(const UShort_t *h, Int_t n)
   { TBuffer::WriteArray((const Short_t *)h, n); }
//______________________________________________________________________________
inline void TBuffer::WriteArray(const UInt_t *i, Int_t n)
   { TBuffer::WriteArray((const Int_t *)i, n); }
//______________________________________________________________________________
inline void TBuffer::WriteArray(const ULong_t *l, Int_t n)
   { TBuffer::WriteArray((const Long_t *)l, n); }

//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const UChar_t *c, Int_t n)
   { TBuffer::WriteFastArray((const Char_t *)c, n); }
//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const UShort_t *h, Int_t n)
   { TBuffer::WriteFastArray((const Short_t *)h, n); }
//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const UInt_t *i, Int_t n)
   { TBuffer::WriteFastArray((const Int_t *)i, n); }
//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const ULong_t *l, Int_t n)
   { TBuffer::WriteFastArray((const Long_t *)l, n); }
# 188 "/cdf/home/pcanal/scratch/code/root.merging/include/TObject.h" 2
# 27 "/cdf/home/pcanal/scratch/code/root.merging/include/TNamed.h" 2


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TList.h" 1
// @(#)root/cont:$Name:  $:$Id: TList.h,v 1.8 2001/03/29 11:25:00 brun Exp $
// Author: Fons Rademakers   10/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TList                                                                //
//                                                                      //
// A doubly linked list. All classes inheriting from TObject can be     //
// inserted in a TList.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TSeqCollection.h" 1
// @(#)root/cont:$Name:  $:$Id: TSeqCollection.h,v 1.5 2001/01/09 18:33:59 rdm Exp $
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSeqCollection                                                       //
//                                                                      //
// Sequenceable collection abstract base class. TSeqCollection's have   //
// an ordering relation, i.e. there is a first and last element.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TCollection.h" 1
// @(#)root/cont:$Name:  $:$Id: TCollection.h,v 1.9 2001/07/05 16:50:50 rdm Exp $
// Author: Fons Rademakers   13/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCollection                                                          //
//                                                                      //
// Collection abstract base class. This class inherits from TObject     //
// because we want to be able to have collections of collections.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////






# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TIterator.h" 1
// @(#)root/cont:$Name:  $:$Id: TIterator.h,v 1.1.1.1 2000/05/16 17:00:40 rdm Exp $
// Author: Fons Rademakers   13/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TIterator                                                            //
//                                                                      //
// Iterator abstract base class. This base class provides the interface //
// for collection iterators.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////





class TCollection;
class TObject;

class TIterator {

protected:
   TIterator() { }
   TIterator(const TIterator &) { }

public:
   virtual TIterator &operator=(const TIterator &) { return *this; }
   virtual ~TIterator() { }
   virtual const TCollection *GetCollection() const = 0;
   virtual Option_t *GetOption() const { return ""; }
   virtual TObject *Next() = 0;
   virtual void Reset() = 0;
   TObject *operator()() { return Next(); }

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 0; } static void Dictionary(); virtual TClass *IsA() const { return TIterator::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TIterator::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TIterator.h"; } static int DeclFileLine() { return 47; } static const char *ImplFileName(); static int ImplFileLine(); //Iterator abstract base class
};
# 31 "/cdf/home/pcanal/scratch/code/root.merging/include/TCollection.h" 2



# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TString.h" 1
// @(#)root/base:$Name:  $:$Id: TString.h,v 1.15.4.1 2002/02/09 16:23:30 rdm Exp $
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TString                                                              //
//                                                                      //
// Basic string class.                                                  //
//                                                                      //
// Cannot be stored in a TCollection... use TObjString instead.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////






# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TMath.h" 1
// @(#)root/base:$Name:  $:$Id: TMath.h,v 1.16 2002/03/29 18:02:47 brun Exp $
// Author: Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMath                                                                //
//                                                                      //
// Encapsulate math routines. For the time being avoid templates.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////





class TMath {

private:
   static Double_t GamCf(Double_t a,Double_t x);
   static Double_t GamSer(Double_t a,Double_t x);

public:

   static Double_t Pi() { return 3.14159265358979323846; }
   static Double_t E() { return 2.7182818284590452354; }

   // Trigo
   static Double_t Sin(Double_t);
   static Double_t Cos(Double_t);
   static Double_t Tan(Double_t);
   static Double_t SinH(Double_t);
   static Double_t CosH(Double_t);
   static Double_t TanH(Double_t);
   static Double_t ASin(Double_t);
   static Double_t ACos(Double_t);
   static Double_t ATan(Double_t);
   static Double_t ATan2(Double_t, Double_t);
   static Double_t ASinH(Double_t);
   static Double_t ACosH(Double_t);
   static Double_t ATanH(Double_t);
   static Double_t Hypot(Double_t x, Double_t y);

   // Misc
   static Double_t Sqrt(Double_t x);
   static Double_t Ceil(Double_t x);
   static Double_t Floor(Double_t x);
   static Double_t Exp(Double_t);
   static Double_t Power(Double_t x, Double_t y);
   static Double_t Log(Double_t x);
   static Double_t Log2(Double_t x);
   static Double_t Log10(Double_t x);
   static Int_t Nint(Float_t x);
   static Int_t Nint(Double_t x);
   static Int_t Finite(Double_t x);
   static Int_t IsNaN(Double_t x);

   // Some integer math
   static Long_t NextPrime(Long_t x); // Least prime number greater than x
   static Long_t Sqrt(Long_t x);
   static Long_t Hypot(Long_t x, Long_t y); // sqrt(px*px + py*py)

   // Abs
   static Short_t Abs(Short_t d);
   static Int_t Abs(Int_t d);
   static Long_t Abs(Long_t d);
   static Float_t Abs(Float_t d);
   static Double_t Abs(Double_t d);

   // Even/Odd
   static Bool_t Even(Long_t a);
   static Bool_t Odd(Long_t a);

   // Sign
   static Short_t Sign(Short_t a, Short_t b);
   static Int_t Sign(Int_t a, Int_t b);
   static Long_t Sign(Long_t a, Long_t b);
   static Float_t Sign(Float_t a, Float_t b);
   static Double_t Sign(Double_t a, Double_t b);

   // Min
   static Short_t Min(Short_t a, Short_t b);
   static UShort_t Min(UShort_t a, UShort_t b);
   static Int_t Min(Int_t a, Int_t b);
   static UInt_t Min(UInt_t a, UInt_t b);
   static Long_t Min(Long_t a, Long_t b);
   static ULong_t Min(ULong_t a, ULong_t b);
   static Float_t Min(Float_t a, Float_t b);
   static Double_t Min(Double_t a, Double_t b);

   // Max
   static Short_t Max(Short_t a, Short_t b);
   static UShort_t Max(UShort_t a, UShort_t b);
   static Int_t Max(Int_t a, Int_t b);
   static UInt_t Max(UInt_t a, UInt_t b);
   static Long_t Max(Long_t a, Long_t b);
   static ULong_t Max(ULong_t a, ULong_t b);
   static Float_t Max(Float_t a, Float_t b);
   static Double_t Max(Double_t a, Double_t b);

   // Locate Min, Max
   static Int_t LocMin(Int_t n, const Short_t *a);
   static Int_t LocMin(Int_t n, const Int_t *a);
   static Int_t LocMin(Int_t n, const Float_t *a);
   static Int_t LocMin(Int_t n, const Double_t *a);
   static Int_t LocMin(Int_t n, const Long_t *a);
   static Int_t LocMax(Int_t n, const Short_t *a);
   static Int_t LocMax(Int_t n, const Int_t *a);
   static Int_t LocMax(Int_t n, const Float_t *a);
   static Int_t LocMax(Int_t n, const Double_t *a);
   static Int_t LocMax(Int_t n, const Long_t *a);

   // Range
   static Short_t Range(Short_t lb, Short_t ub, Short_t x);
   static Int_t Range(Int_t lb, Int_t ub, Int_t x);
   static Long_t Range(Long_t lb, Long_t ub, Long_t x);
   static ULong_t Range(ULong_t lb, ULong_t ub, ULong_t x);
   static Double_t Range(Double_t lb, Double_t ub, Double_t x);

   // Binary search
   static Int_t BinarySearch(Int_t n, const Short_t *array, Short_t value);
   static Int_t BinarySearch(Int_t n, const Short_t **array, Short_t value);
   static Int_t BinarySearch(Int_t n, const Int_t *array, Int_t value);
   static Int_t BinarySearch(Int_t n, const Int_t **array, Int_t value);
   static Int_t BinarySearch(Int_t n, const Float_t *array, Float_t value);
   static Int_t BinarySearch(Int_t n, const Float_t **array, Float_t value);
   static Int_t BinarySearch(Int_t n, const Double_t *array, Double_t value);
   static Int_t BinarySearch(Int_t n, const Double_t **array, Double_t value);
   static Int_t BinarySearch(Int_t n, const Long_t *array, Long_t value);
   static Int_t BinarySearch(Int_t n, const Long_t **array, Long_t value);

   // Hashing
   static ULong_t Hash(const void *txt, Int_t ntxt);
   static ULong_t Hash(const char *str);

   // Sorting
   static void Sort(Int_t n, const Short_t *a, Int_t *index, Bool_t down=kTRUE);
   static void Sort(Int_t n, const Int_t *a, Int_t *index, Bool_t down=kTRUE);
   static void Sort(Int_t n, const Float_t *a, Int_t *index, Bool_t down=kTRUE);
   static void Sort(Int_t n, const Double_t *a, Int_t *index, Bool_t down=kTRUE);
   static void Sort(Int_t n, const Long_t *a, Int_t *index, Bool_t down=kTRUE);
   static void BubbleHigh(Int_t Narr, Double_t *arr1, Int_t *arr2);
   static void BubbleLow (Int_t Narr, Double_t *arr1, Int_t *arr2);

   // Advanced
   static Float_t *Cross(Float_t v1[3],Float_t v2[3],Float_t out[3]); // Calculate the Cross Product of two vectors
   static Float_t Normalize(Float_t v[3]); // Normalize a vector
   static Float_t NormCross(Float_t v1[3],Float_t v2[3],Float_t out[3]); // Calculate the Normalized Cross Product of two vectors
   static Float_t *Normal2Plane(Float_t v1[3],Float_t v2[3],Float_t v3[3], Float_t normal[3]); // Calcualte a normal vector of a plane

   static Double_t *Cross(Double_t v1[3],Double_t v2[3],Double_t out[3]);// Calculate the Cross Product of two vectors
   static Double_t Erf(Double_t x);
   static Double_t Erfc(Double_t x);
   static Double_t Freq(Double_t x);
   static Double_t Gamma(Double_t z);
   static Double_t Gamma(Double_t a,Double_t x);
   static Double_t BreitWigner(Double_t x, Double_t mean=0, Double_t gamma=1);
   static Double_t Gaus(Double_t x, Double_t mean=0, Double_t sigma=1);
   static Double_t Landau(Double_t x, Double_t mean=0, Double_t sigma=1);
   static Double_t LnGamma(Double_t z);
   static Double_t Normalize(Double_t v[3]); // Normalize a vector
   static Double_t NormCross(Double_t v1[3],Double_t v2[3],Double_t out[3]); // Calculate the Normalized Cross Product of two vectors
   static Double_t *Normal2Plane(Double_t v1[3],Double_t v2[3],Double_t v3[3], Double_t normal[3]); // Calcualte a normal vector of a plane
   static Double_t Prob(Double_t chi2,Int_t ndf);
   static Double_t KolmogorovProb(Double_t z);

   // Bessel functions
   static Double_t BesselI(Int_t n,Double_t x); // integer order modified Bessel function I_n(x)
   static Double_t BesselK(Int_t n,Double_t x); // integer order modified Bessel function K_n(x)
   static Double_t BesselI0(Double_t x); // modified Bessel function I_0(x)
   static Double_t BesselK0(Double_t x); // modified Bessel function K_0(x)
   static Double_t BesselI1(Double_t x); // modified Bessel function I_1(x)
   static Double_t BesselK1(Double_t x); // modified Bessel function K_1(x)
   static Double_t BesselJ0(Double_t x); // Bessel function J0(x) for any real x
   static Double_t BesselJ1(Double_t x); // Bessel function J1(x) for any real x
   static Double_t BesselY0(Double_t x); // Bessel function Y0(x) for positive x
   static Double_t BesselY1(Double_t x); // Bessel function Y1(x) for positive x
   static Double_t Struve(Int_t n, Double_t x); // Struve functions of order 0 and 1

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 0; } static void Dictionary(); virtual TClass *IsA() const { return TMath::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TMath::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TMath.h"; } static int DeclFileLine() { return 191; } static const char *ImplFileName(); static int ImplFileLine(); //Interface to math routines
};


//---- Even/odd ----------------------------------------------------------------

inline Bool_t TMath::Even(Long_t a)
   { return ! (a & 1); }

inline Bool_t TMath::Odd(Long_t a)
   { return (a & 1); }

//---- Abs ---------------------------------------------------------------------

inline Short_t TMath::Abs(Short_t d)
   { return (d > 0) ? d : -d; }

inline Int_t TMath::Abs(Int_t d)
   { return (d > 0) ? d : -d; }

inline Long_t TMath::Abs(Long_t d)
   { return (d > 0) ? d : -d; }

inline Float_t TMath::Abs(Float_t d)
   { return (d > 0) ? d : -d; }

inline Double_t TMath::Abs(Double_t d)
   { return (d > 0) ? d : -d; }

//---- Sign --------------------------------------------------------------------

inline Short_t TMath::Sign(Short_t a, Short_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Int_t TMath::Sign(Int_t a, Int_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Long_t TMath::Sign(Long_t a, Long_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Float_t TMath::Sign(Float_t a, Float_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Double_t TMath::Sign(Double_t a, Double_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

//---- Min ---------------------------------------------------------------------

inline Short_t TMath::Min(Short_t a, Short_t b)
   { return a <= b ? a : b; }

inline UShort_t TMath::Min(UShort_t a, UShort_t b)
   { return a <= b ? a : b; }

inline Int_t TMath::Min(Int_t a, Int_t b)
   { return a <= b ? a : b; }

inline UInt_t TMath::Min(UInt_t a, UInt_t b)
   { return a <= b ? a : b; }

inline Long_t TMath::Min(Long_t a, Long_t b)
   { return a <= b ? a : b; }

inline ULong_t TMath::Min(ULong_t a, ULong_t b)
   { return a <= b ? a : b; }

inline Float_t TMath::Min(Float_t a, Float_t b)
   { return a <= b ? a : b; }

inline Double_t TMath::Min(Double_t a, Double_t b)
   { return a <= b ? a : b; }

//---- Max ---------------------------------------------------------------------

inline Short_t TMath::Max(Short_t a, Short_t b)
   { return a >= b ? a : b; }

inline UShort_t TMath::Max(UShort_t a, UShort_t b)
   { return a >= b ? a : b; }

inline Int_t TMath::Max(Int_t a, Int_t b)
   { return a >= b ? a : b; }

inline UInt_t TMath::Max(UInt_t a, UInt_t b)
   { return a >= b ? a : b; }

inline Long_t TMath::Max(Long_t a, Long_t b)
   { return a >= b ? a : b; }

inline ULong_t TMath::Max(ULong_t a, ULong_t b)
   { return a >= b ? a : b; }

inline Float_t TMath::Max(Float_t a, Float_t b)
   { return a >= b ? a : b; }

inline Double_t TMath::Max(Double_t a, Double_t b)
   { return a >= b ? a : b; }

//---- Range -------------------------------------------------------------------

inline Short_t TMath::Range(Short_t lb, Short_t ub, Short_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Int_t TMath::Range(Int_t lb, Int_t ub, Int_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Long_t TMath::Range(Long_t lb, Long_t ub, Long_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline ULong_t TMath::Range(ULong_t lb, ULong_t ub, ULong_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Double_t TMath::Range(Double_t lb, Double_t ub, Double_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

//---- Trig and other functions ------------------------------------------------


# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/float.h" 1
# 310 "/cdf/home/pcanal/scratch/code/root.merging/include/TMath.h" 2
# 323 "/cdf/home/pcanal/scratch/code/root.merging/include/TMath.h"
// math functions are defined inline so we have to include them here
# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/math.h" 1





#pragma include_noerr <stdfunc.dll>
# 325 "/cdf/home/pcanal/scratch/code/root.merging/include/TMath.h" 2




// don't want to include complete <math.h>
# 357 "/cdf/home/pcanal/scratch/code/root.merging/include/TMath.h"
inline Double_t TMath::Sin(Double_t x)
   { return sin(x); }

inline Double_t TMath::Cos(Double_t x)
   { return cos(x); }

inline Double_t TMath::Tan(Double_t x)
   { return tan(x); }

inline Double_t TMath::SinH(Double_t x)
   { return sinh(x); }

inline Double_t TMath::CosH(Double_t x)
   { return cosh(x); }

inline Double_t TMath::TanH(Double_t x)
   { return tanh(x); }

inline Double_t TMath::ASin(Double_t x)
   { return asin(x); }

inline Double_t TMath::ACos(Double_t x)
   { return acos(x); }

inline Double_t TMath::ATan(Double_t x)
   { return atan(x); }

inline Double_t TMath::ATan2(Double_t y, Double_t x)
   { return x != 0 ? atan2(y, x) : (y > 0 ? Pi()/2 : -Pi()/2); }

inline Double_t TMath::Sqrt(Double_t x)
   { return sqrt(x); }

inline Double_t TMath::Exp(Double_t x)
   { return exp(x); }

inline Double_t TMath::Power(Double_t x, Double_t y)
   { return pow(x, y); }

inline Double_t TMath::Log(Double_t x)
   { return log(x); }

inline Double_t TMath::Log10(Double_t x)
   { return log10(x); }

inline Int_t TMath::Finite(Double_t x)



   { return finite(x); }


inline Int_t TMath::IsNaN(Double_t x)
   { return isnan(x); }

//-------- Advanced -------------

inline Float_t TMath::NormCross(Float_t v1[3],Float_t v2[3],Float_t out[3])
{
   // Calculate the Normalized Cross Product of two vectors
   return Normalize(Cross(v1,v2,out));
}

inline Double_t TMath::NormCross(Double_t v1[3],Double_t v2[3],Double_t out[3])
{
   // Calculate the Normalized Cross Product of two vectors
   return Normalize(Cross(v1,v2,out));
}
# 32 "/cdf/home/pcanal/scratch/code/root.merging/include/TString.h" 2



# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TRefCnt.h" 1
// @(#)root/base:$Name:  $:$Id: TRefCnt.h,v 1.2 2000/12/13 16:45:35 brun Exp $
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TRefCnt                                                             //
//                                                                      //
//  Base class for reference counted objects.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////






class TRefCnt {

protected:
   UInt_t fRefs; // (1 less than) number of references

public:
   enum EReferenceFlag { kStaticInit };

   TRefCnt(Int_t initRef = 0) : fRefs((UInt_t)initRef-1) { }
   TRefCnt(EReferenceFlag) { } // leave fRefs alone
   UInt_t References() const { return fRefs+1; }
   void SetRefCount(UInt_t r) { fRefs = r-1; }
   void AddReference() { fRefs++; }
   UInt_t RemoveReference() { return fRefs--; }
};
# 36 "/cdf/home/pcanal/scratch/code/root.merging/include/TString.h" 2
# 48 "/cdf/home/pcanal/scratch/code/root.merging/include/TString.h"
class TRegexp;
class TString;
class TSubString;

TString operator+(const TString& s1, const TString& s2);
TString operator+(const TString& s, const char *cs);
TString operator+(const char *cs, const TString& s);
TString operator+(const TString& s, char c);
TString operator+(const TString& s, Long_t i);
TString operator+(const TString& s, ULong_t i);
TString operator+(char c, const TString& s);
TString operator+(Long_t i, const TString& s);
TString operator+(ULong_t i, const TString& s);
Bool_t operator==(const TString& s1, const TString& s2);
Bool_t operator==(const TString& s1, const char *s2);
Bool_t operator==(const TSubString& s1, const TSubString& s2);
Bool_t operator==(const TSubString& s1, const TString& s2);
Bool_t operator==(const TSubString& s1, const char *s2);


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TStringRef                                                          //
//                                                                      //
//  This is the dynamically allocated part of a TString.                //
//  It maintains a reference count. It contains no public member        //
//  functions.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TStringRef : public TRefCnt {

friend class TString;
friend class TStringLong;
friend class TSubString;

private:
   Ssiz_t fCapacity; // Max string length (excluding null)
   Ssiz_t fNchars; // String length (excluding null)

   void UnLink(); // disconnect from a TStringRef, maybe delete it

   Ssiz_t Length() const { return fNchars; }
   Ssiz_t Capacity() const { return fCapacity; }
   char *Data() const { return (char*)(this+1); }

   char& operator[](Ssiz_t i) { return ((char*)(this+1))[i]; }
   char operator[](Ssiz_t i) const { return ((char*)(this+1))[i]; }

   Ssiz_t First(char c) const;
   Ssiz_t First(const char *s) const;
   unsigned Hash() const;
   unsigned HashFoldCase() const;
   Ssiz_t Last(char) const;

   static TStringRef *GetRep(Ssiz_t capac, Ssiz_t nchar);
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TSubString                                                          //
//                                                                      //
//  The TSubString class allows selected elements to be addressed.      //
//  There are no public constructors.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TSubString {

friend class TStringLong;
friend class TString;

friend Bool_t operator==(const TSubString& s1, const TSubString& s2);
friend Bool_t operator==(const TSubString& s1, const TString& s2);
friend Bool_t operator==(const TSubString& s1, const char *s2);

private:
   TString *fStr; // Referenced string
   Ssiz_t fBegin; // Index of starting character
   Ssiz_t fExtent; // Length of TSubString

   // NB: the only constructor is private
   TSubString(const TString& s, Ssiz_t start, Ssiz_t len);

protected:
   void SubStringError(Ssiz_t, Ssiz_t, Ssiz_t) const;
   void AssertElement(Ssiz_t i) const; // Verifies i is valid index

public:
   TSubString(const TSubString& s)
     : fStr(s.fStr), fBegin(s.fBegin), fExtent(s.fExtent) { }

   TSubString& operator=(const char *s); // Assignment to char*
   TSubString& operator=(const TString& s); // Assignment to TString
   char& operator()(Ssiz_t i); // Index with optional bounds checking
   char& operator[](Ssiz_t i); // Index with bounds checking
   char operator()(Ssiz_t i) const; // Index with optional bounds checking
   char operator[](Ssiz_t i) const; // Index with bounds checking

   const char *Data() const;
   Ssiz_t Length() const { return fExtent; }
   Ssiz_t Start() const { return fBegin; }
   void ToLower(); // Convert self to lower-case
   void ToUpper(); // Convert self to upper-case

   // For detecting null substrings
   Bool_t IsNull() const { return fBegin == kNPOS; }
   int operator!() const { return fBegin == kNPOS; }
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TString                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TString {

friend class TSubString;
friend class TStringRef;

friend TString operator+(const TString& s1, const TString& s2);
friend TString operator+(const TString& s, const char *cs);
friend TString operator+(const char *cs, const TString& s);
friend TString operator+(const TString& s, char c);
friend TString operator+(const TString& s, Long_t i);
friend TString operator+(const TString& s, ULong_t i);
friend TString operator+(char c, const TString& s);
friend TString operator+(Long_t i, const TString& s);
friend TString operator+(ULong_t i, const TString& s);
friend Bool_t operator==(const TString& s1, const TString& s2);
friend Bool_t operator==(const TString& s1, const char *s2);

private:
   static Ssiz_t fgInitialCapac; // Initial allocation Capacity
   static Ssiz_t fgResizeInc; // Resizing increment
   static Ssiz_t fgFreeboard; // Max empty space before reclaim

   void Clone(); // Make self a distinct copy
   void Clone(Ssiz_t nc); // Make self a distinct copy w. capacity nc

protected:
   char *fData; // ref. counted data (TStringRef is in front)

   // Special concatenation constructor
   TString(const char *a1, Ssiz_t n1, const char *a2, Ssiz_t n2);
   TStringRef *Pref() const { return (((TStringRef*) fData) - 1); }
   void AssertElement(Ssiz_t nc) const; // Index in range
   void Clobber(Ssiz_t nc); // Remove old contents
   void Cow(); // Do copy on write as needed
   void Cow(Ssiz_t nc); // Do copy on write as needed
   static Ssiz_t AdjustCapacity(Ssiz_t nc);
   void InitChar(char c); // Initialize from char

public:
   enum EStripType { kLeading = 0x1, kTrailing = 0x2, kBoth = 0x3 };
   enum ECaseCompare { kExact, kIgnoreCase };

   TString(); // Null string
   TString(Ssiz_t ic); // Suggested capacity
   TString(const TString& s) // Copy constructor
      { fData = s.fData; Pref()->AddReference(); }

   TString(const char *s); // Copy to embedded null
   TString(const char *s, Ssiz_t n); // Copy past any embedded nulls
   TString(char c) { InitChar(c); }

   TString(char c, Ssiz_t s);

   TString(const TSubString& sub);

   virtual ~TString();

   // ROOT I/O interface
   virtual void FillBuffer(char *&buffer);
   virtual void ReadBuffer(char *&buffer);
   virtual Int_t Sizeof() const;

   static TString *ReadString(TBuffer &b, const TClass *clReq);
   static void WriteString(TBuffer &b, const TString *a);

   friend TBuffer &operator<<(TBuffer &b, const TString *obj);

   // Type conversion
   operator const char*() const { return fData; }

   // Assignment
   TString& operator=(char s); // Replace string
   TString& operator=(const char *s);
   TString& operator=(const TString& s);
   TString& operator=(const TSubString& s);
   TString& operator+=(const char *s); // Append string
   TString& operator+=(const TString& s);
   TString& operator+=(char c);
   TString& operator+=(Short_t i);
   TString& operator+=(UShort_t i);
   TString& operator+=(Int_t i);
   TString& operator+=(UInt_t i);
   TString& operator+=(Long_t i);
   TString& operator+=(ULong_t i);
   TString& operator+=(Float_t f);
   TString& operator+=(Double_t f);

   // Indexing operators
   char& operator[](Ssiz_t i); // Indexing with bounds checking
   char& operator()(Ssiz_t i); // Indexing with optional bounds checking
   TSubString operator()(Ssiz_t start, Ssiz_t len); // Sub-string operator
   TSubString operator()(const TRegexp& re); // Match the RE
   TSubString operator()(const TRegexp& re, Ssiz_t start);
   TSubString SubString(const char *pat, Ssiz_t start = 0,
                           ECaseCompare cmp = kExact);
   char operator[](Ssiz_t i) const;
   char operator()(Ssiz_t i) const;
   TSubString operator()(Ssiz_t start, Ssiz_t len) const;
   TSubString operator()(const TRegexp& re) const; // Match the RE
   TSubString operator()(const TRegexp& re, Ssiz_t start) const;
   TSubString SubString(const char *pat, Ssiz_t start = 0,
                           ECaseCompare cmp = kExact) const;

   // Non-static member functions
   TString& Append(const char *cs);
   TString& Append(const char *cs, Ssiz_t n);
   TString& Append(const TString& s);
   TString& Append(const TString& s, Ssiz_t n);
   TString& Append(char c, Ssiz_t rep = 1); // Append c rep times
   Bool_t BeginsWith(const char *s, ECaseCompare cmp = kExact) const;
   Bool_t BeginsWith(const TString& pat, ECaseCompare cmp = kExact) const;
   Ssiz_t Capacity() const { return Pref()->Capacity(); }
   Ssiz_t Capacity(Ssiz_t n);
   TString& Chop();
   int CompareTo(const char *cs, ECaseCompare cmp = kExact) const;
   int CompareTo(const TString& st, ECaseCompare cmp = kExact) const;
   Bool_t Contains(const char *pat, ECaseCompare cmp = kExact) const;
   Bool_t Contains(const TString& pat, ECaseCompare cmp = kExact) const;
   TString Copy() const;
   const char *Data() const { return fData; }
   Bool_t EndsWith(const char *pat, ECaseCompare cmp = kExact) const;
   Ssiz_t First(char c) const { return Pref()->First(c); }
   Ssiz_t First(const char *cs) const { return Pref()->First(cs); }
   unsigned Hash(ECaseCompare cmp = kExact) const;
   Ssiz_t Index(const char *pat, Ssiz_t i = 0,
                      ECaseCompare cmp = kExact) const;
   Ssiz_t Index(const TString& s, Ssiz_t i = 0,
                      ECaseCompare cmp = kExact) const;
   Ssiz_t Index(const char *pat, Ssiz_t patlen, Ssiz_t i,
                      ECaseCompare cmp) const;
   Ssiz_t Index(const TString& s, Ssiz_t patlen, Ssiz_t i,
                      ECaseCompare cmp) const;
   Ssiz_t Index(const TRegexp& pat, Ssiz_t i = 0) const;
   Ssiz_t Index(const TRegexp& pat, Ssiz_t *ext, Ssiz_t i = 0) const;
   TString& Insert(Ssiz_t pos, const char *s);
   TString& Insert(Ssiz_t pos, const char *s, Ssiz_t extent);
   TString& Insert(Ssiz_t pos, const TString& s);
   TString& Insert(Ssiz_t pos, const TString& s, Ssiz_t extent);
   Bool_t IsAscii() const;
   Bool_t IsNull() const { return Pref()->fNchars == 0; }
   Ssiz_t Last(char c) const { return Pref()->Last(c); }
   Ssiz_t Length() const { return Pref()->fNchars; }
   TString& Prepend(const char *cs); // Prepend a character string
   TString& Prepend(const char *cs, Ssiz_t n);
   TString& Prepend(const TString& s);
   TString& Prepend(const TString& s, Ssiz_t n);
   TString& Prepend(char c, Ssiz_t rep = 1); // Prepend c rep times
   istream& ReadFile(istream& str); // Read to EOF or null character
   istream& ReadLine(istream& str,
                         Bool_t skipWhite = kTRUE); // Read to EOF or newline
   istream& ReadString(istream& str); // Read to EOF or null character
   istream& ReadToDelim(istream& str, char delim = '\n'); // Read to EOF or delimitor
   istream& ReadToken(istream& str); // Read separated by white space
   TString& Remove(Ssiz_t pos); // Remove pos to end of string
   TString& Remove(Ssiz_t pos, Ssiz_t n); // Remove n chars starting at pos
   TString& Replace(Ssiz_t pos, Ssiz_t n, const char *s);
   TString& Replace(Ssiz_t pos, Ssiz_t n, const char *s, Ssiz_t ns);
   TString& Replace(Ssiz_t pos, Ssiz_t n, const TString& s);
   TString& Replace(Ssiz_t pos, Ssiz_t n1, const TString& s, Ssiz_t n2);
   TString& ReplaceAll(const TString& s1, const TString& s2); // Find&Replace all s1 with s2 if any
   TString& ReplaceAll(const TString& s1, const char *s2); // Find&Replace all s1 with s2 if any
   TString& ReplaceAll(const char *s1, const TString& s2); // Find&Replace all s1 with s2 if any
   TString& ReplaceAll(const char *s1, const char *s2); // Find&Replace all s1 with s2 if any
   TString& ReplaceAll(const char *s1, Ssiz_t ls1, const char *s2, Ssiz_t ls2); // Find&Replace all s1 with s2 if any
   void Resize(Ssiz_t n); // Truncate or add blanks as necessary
   TSubString Strip(EStripType s = kTrailing, char c = ' ');
   TSubString Strip(EStripType s = kTrailing, char c = ' ') const;
   void ToLower(); // Change self to lower-case
   void ToUpper(); // Change self to upper-case

   // Static member functions
   static Ssiz_t InitialCapacity(Ssiz_t ic = 15); // Initial allocation capacity
   static Ssiz_t MaxWaste(Ssiz_t mw = 15); // Max empty space before reclaim
   static Ssiz_t ResizeIncrement(Ssiz_t ri = 16); // Resizing increment
   static Ssiz_t GetInitialCapacity();
   static Ssiz_t GetResizeIncrement();
   static Ssiz_t GetMaxWaste();

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 1; } static void Dictionary(); virtual TClass *IsA() const { return TString::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TString::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TString.h"; } static int DeclFileLine() { return 341; } static const char *ImplFileName(); static int ImplFileLine(); //Basic string class
};

// Related global functions
istream& operator>>(istream& str, TString& s);
ostream& operator<<(ostream& str, const TString& s);
TBuffer& operator>>(TBuffer& buf, TString& s);
TBuffer& operator<<(TBuffer& buf, const TString& s);
TBuffer& operator>>(TBuffer& buf, TString*& sp);

TString ToLower(const TString&); // Return lower-case version of argument
TString ToUpper(const TString&); // Return upper-case version of argument
inline unsigned Hash(const TString& s) { return s.Hash(); }
inline unsigned Hash(const TString *s) { return s->Hash(); }
        unsigned Hash(const char *s);

extern char *Form(const char *fmt, ...); // format in circular buffer
extern void Printf(const char *fmt, ...); // format and print
extern char *Strip(const char *str, char c = ' '); // strip c off str, free with delete []
extern char *StrDup(const char *str); // duplicate str, free with delete []
extern char *Compress(const char *str); // remove blanks from string, free with delele []
extern int EscChar(const char *src, char *dst, int dstlen, char *specchars,
                     char escchar); // copy from src to dst escaping specchars by escchar
extern int UnEscChar(const char *src, char *dst, int dstlen, char *specchars,
                     char escchar); // copy from src to dst removing escchar from specchars







//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Inlines                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

inline void TStringRef::UnLink()
{ if (RemoveReference() == 0) delete [] (char*)this; }

inline void TString::Cow()
{ if (Pref()->References() > 1) Clone(); }

inline void TString::Cow(Ssiz_t nc)
{ if (Pref()->References() > 1 || Capacity() < nc) Clone(nc); }

inline TString& TString::Append(const char *cs)
{ return Replace(Length(), 0, cs, strlen(cs)); }

inline TString& TString::Append(const char* cs, Ssiz_t n)
{ return Replace(Length(), 0, cs, n); }

inline TString& TString::Append(const TString& s)
{ return Replace(Length(), 0, s.Data(), s.Length()); }

inline TString& TString::Append(const TString& s, Ssiz_t n)
{ return Replace(Length(), 0, s.Data(), TMath::Min(n, s.Length())); }

inline TString& TString::operator+=(const char* cs)
{ return Append(cs, strlen(cs)); }

inline TString& TString::operator+=(const TString& s)
{ return Append(s.Data(), s.Length()); }

inline TString& TString::operator+=(char c)
{ return Append(c); }

inline TString& TString::operator+=(Long_t i)
{ return operator+=(Form("%ld", i)); }

inline TString& TString::operator+=(ULong_t i)
{ return operator+=(Form("%lu", i)); }

inline TString& TString::operator+=(Short_t i)
{ return operator+=((Long_t) i); }

inline TString& TString::operator+=(UShort_t i)
{ return operator+=((ULong_t) i); }

inline TString& TString::operator+=(Int_t i)
{ return operator+=((Long_t) i); }

inline TString& TString::operator+=(UInt_t i)
{ return operator+=((ULong_t) i); }

inline TString& TString::operator+=(Double_t f)
{ return operator+=(Form("%9.9g", f)); }

inline TString& TString::operator+=(Float_t f)
{ return operator+=((Double_t) f); }

inline Bool_t TString::BeginsWith(const char* s, ECaseCompare cmp) const
{ return Index(s, strlen(s), (Ssiz_t)0, cmp) == 0; }

inline Bool_t TString::BeginsWith(const TString& pat, ECaseCompare cmp) const
{ return Index(pat.Data(), pat.Length(), (Ssiz_t)0, cmp) == 0; }

inline Bool_t TString::Contains(const TString& pat, ECaseCompare cmp) const
{ return Index(pat.Data(), pat.Length(), (Ssiz_t)0, cmp) != kNPOS; }

inline Bool_t TString::Contains(const char* s, ECaseCompare cmp) const
{ return Index(s, strlen(s), (Ssiz_t)0, cmp) != kNPOS; }

inline Ssiz_t TString::Index(const char* s, Ssiz_t i, ECaseCompare cmp) const
{ return Index(s, strlen(s), i, cmp); }

inline Ssiz_t TString::Index(const TString& s, Ssiz_t i, ECaseCompare cmp) const
{ return Index(s.Data(), s.Length(), i, cmp); }

inline Ssiz_t TString::Index(const TString& pat, Ssiz_t patlen, Ssiz_t i,
                             ECaseCompare cmp) const
{ return Index(pat.Data(), patlen, i, cmp); }

inline TString& TString::Insert(Ssiz_t pos, const char* cs)
{ return Replace(pos, 0, cs, strlen(cs)); }

inline TString& TString::Insert(Ssiz_t pos, const char* cs, Ssiz_t n)
{ return Replace(pos, 0, cs, n); }

inline TString& TString::Insert(Ssiz_t pos, const TString& s)
{ return Replace(pos, 0, s.Data(), s.Length()); }

inline TString& TString::Insert(Ssiz_t pos, const TString& s, Ssiz_t n)
{ return Replace(pos, 0, s.Data(), TMath::Min(n, s.Length())); }

inline TString& TString::Prepend(const char* cs)
{ return Replace(0, 0, cs, strlen(cs)); }

inline TString& TString::Prepend(const char* cs, Ssiz_t n)
{ return Replace(0, 0, cs, n); }

inline TString& TString::Prepend(const TString& s)
{ return Replace(0, 0, s.Data(), s.Length()); }

inline TString& TString::Prepend(const TString& s, Ssiz_t n)
{ return Replace(0, 0, s.Data(), TMath::Min(n, s.Length())); }

inline TString& TString::Remove(Ssiz_t pos)
{ return Replace(pos, TMath::Max(0, Length()-pos), 0, 0); }

inline TString& TString::Remove(Ssiz_t pos, Ssiz_t n)
{ return Replace(pos, n, 0, 0); }

inline TString& TString::Chop()
{ return Remove(TMath::Max(0,Length()-1)); }

inline TString& TString::Replace(Ssiz_t pos, Ssiz_t n, const char* cs)
{ return Replace(pos, n, cs, strlen(cs)); }

inline TString& TString::Replace(Ssiz_t pos, Ssiz_t n, const TString& s)
{ return Replace(pos, n, s.Data(), s.Length()); }

inline TString& TString::Replace(Ssiz_t pos, Ssiz_t n1, const TString& s,
                                 Ssiz_t n2)
{ return Replace(pos, n1, s.Data(), TMath::Min(s.Length(), n2)); }

inline TString& TString::ReplaceAll(const TString& s1,const TString& s2)
{ return ReplaceAll( s1.Data(), s1.Length(), s2.Data(), s2.Length()) ; }

inline TString& TString::ReplaceAll(const TString& s1,const char *s2)
{ return ReplaceAll( s1.Data(), s1.Length(), s2, s2 ? strlen(s2):0) ; }

inline TString& TString::ReplaceAll(const char *s1,const TString& s2)
{ return ReplaceAll( s1, s1 ? strlen(s1): 0, s2.Data(), s2.Length()) ; }

inline TString& TString::ReplaceAll(const char *s1,const char *s2)
{ return ReplaceAll( s1, s1?strlen(s1):0, s2, s2?strlen(s2):0) ; }

inline char& TString::operator()(Ssiz_t i)
{ Cow(); return fData[i]; }

inline char TString::operator[](Ssiz_t i) const
{ AssertElement(i); return fData[i]; }

inline char TString::operator()(Ssiz_t i) const
{ return fData[i]; }

inline const char* TSubString::Data() const
{ return fStr->Data() + fBegin; }

// Access to elements of sub-string with bounds checking
inline char TSubString::operator[](Ssiz_t i) const
{ AssertElement(i); return fStr->fData[fBegin+i]; }

inline char TSubString::operator()(Ssiz_t i) const
{ return fStr->fData[fBegin+i]; }

// String Logical operators

inline Bool_t operator==(const TString& s1, const TString& s2)
{
   return ((s1.Length() == s2.Length()) &&
            !memcmp(s1.Data(), s2.Data(), s1.Length()));
}


inline Bool_t operator!=(const TString& s1, const TString& s2)
{ return !(s1 == s2); }

inline Bool_t operator< (const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)< 0; }

inline Bool_t operator> (const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)> 0; }

inline Bool_t operator<=(const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)<=0; }

inline Bool_t operator>=(const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)>=0; }

//     Bool_t     operator==(const TString& s1, const char* s2);
inline Bool_t operator!=(const TString& s1, const char* s2)
{ return !(s1 == s2); }

inline Bool_t operator< (const TString& s1, const char* s2)
{ return s1.CompareTo(s2)< 0; }

inline Bool_t operator> (const TString& s1, const char* s2)
{ return s1.CompareTo(s2)> 0; }

inline Bool_t operator<=(const TString& s1, const char* s2)
{ return s1.CompareTo(s2)<=0; }

inline Bool_t operator>=(const TString& s1, const char* s2)
{ return s1.CompareTo(s2)>=0; }

inline Bool_t operator==(const char* s1, const TString& s2)
{ return (s2 == s1); }

inline Bool_t operator!=(const char* s1, const TString& s2)
{ return !(s2 == s1); }

inline Bool_t operator< (const char* s1, const TString& s2)
{ return s2.CompareTo(s1)> 0; }

inline Bool_t operator> (const char* s1, const TString& s2)
{ return s2.CompareTo(s1)< 0; }

inline Bool_t operator<=(const char* s1, const TString& s2)
{ return s2.CompareTo(s1)>=0; }

inline Bool_t operator>=(const char* s1, const TString& s2)
{ return s2.CompareTo(s1)<=0; }

// SubString Logical operators
//     Bool_t     operator==(const TSubString& s1, const TSubString& s2);
//     Bool_t     operator==(const TSubString& s1, const char* s2);
//     Bool_t     operator==(const TSubString& s1, const TString& s2);
inline Bool_t operator==(const TString& s1, const TSubString& s2)
{ return (s2 == s1); }

inline Bool_t operator==(const char* s1, const TSubString& s2)
{ return (s2 == s1); }

inline Bool_t operator!=(const TSubString& s1, const char* s2)
{ return !(s1 == s2); }

inline Bool_t operator!=(const TSubString& s1, const TString& s2)
{ return !(s1 == s2); }

inline Bool_t operator!=(const TSubString& s1, const TSubString& s2)
{ return !(s1 == s2); }

inline Bool_t operator!=(const TString& s1, const TSubString& s2)
{ return !(s2 == s1); }

inline Bool_t operator!=(const char* s1, const TSubString& s2)
{ return !(s2 == s1); }
# 35 "/cdf/home/pcanal/scratch/code/root.merging/include/TCollection.h" 2


class TClass;
class TObjectTable;


const Bool_t kIterForward = kTRUE;
const Bool_t kIterBackward = !kIterForward;


class TCollection : public TObject {

private:
   static TCollection *fgCurrentCollection; //used by macro ForEach
   static TObjectTable *fgGarbageCollection; //used by garbage collector
   static Bool_t fgEmptyingGarbage; //used by garbage collector
   static Int_t fgGarbageStack; //used by garbage collector

   TCollection(const TCollection &); // private and not-implemented, collections
   void operator=(const TCollection &); // are too sensitive to be automatically copied

protected:
   enum { kIsOwner = (1 << (14)) };

   TString fName; //name of the collection
   Int_t fSize; //number of elements in collection

   TCollection() : fSize(0) { }

public:
   enum { kInitCapacity = 16, kInitHashTableCapacity = 17 };

   virtual ~TCollection() { }
   virtual void Add(TObject *obj) = 0;
   void AddVector(TObject *obj1, ...);
   virtual void AddAll(TCollection *col);
   Bool_t AssertClass(TClass *cl) const;
   void Browse(TBrowser *b);
   Int_t Capacity() const { return fSize; }
   virtual void Clear(Option_t *option="") = 0;
   Bool_t Contains(const char *name) const { return FindObject(name) != 0; }
   Bool_t Contains(const TObject *obj) const { return FindObject(obj) != 0; }
   virtual void Delete(Option_t *option="") = 0;
   virtual void Draw(Option_t *option="");
   virtual void Dump() const ;
   virtual TObject *FindObject(const char *name) const;
   TObject *operator()(const char *name) const;
   virtual TObject *FindObject(const TObject *obj) const;
   virtual const char *GetName() const;
   virtual TObject **GetObjectRef(TObject *obj) const = 0;
   virtual Int_t GetSize() const { return fSize; }
   virtual Int_t GrowBy(Int_t delta) const;
   Bool_t IsArgNull(const char *where, const TObject *obj) const;
   virtual Bool_t IsEmpty() const { return GetSize() <= 0; }
   Bool_t IsFolder() const { return kTRUE; }
   Bool_t IsOwner() const { return TestBit(kIsOwner); }
   virtual void ls(Option_t *option="") const ;
   virtual TIterator *MakeIterator(Bool_t dir = kIterForward) const = 0;
   virtual TIterator *MakeReverseIterator() const { return MakeIterator(kIterBackward); }
   virtual void Paint(Option_t *option="");
   virtual void Print(Option_t *option="") const;
   virtual void RecursiveRemove(TObject *obj);
   virtual TObject *Remove(TObject *obj) = 0;
   virtual void RemoveAll(TCollection *col);
   void RemoveAll() { Clear(); }
   void SetCurrentCollection();
   void SetName(const char *name) { fName = name; }
   void SetOwner(Bool_t enable = kTRUE) { enable ? SetBit(kIsOwner) : ResetBit(kIsOwner); }
   virtual Int_t Write(const char *name=0, Int_t option=0, Int_t bufsize=0);

   static TCollection *GetCurrentCollection();
   static void StartGarbageCollection();
   static void GarbageCollect(TObject *obj);
   static void EmptyGarbageCollection();

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 3; } static void Dictionary(); virtual TClass *IsA() const { return TCollection::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TCollection::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TCollection.h"; } static int DeclFileLine() { return 110; } static const char *ImplFileName(); static int ImplFileLine(); //Collection abstract base class
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TIter                                                                //
//                                                                      //
// Iterator wrapper. Type of iterator used depends on type of           //
// collection.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TIter {

private:
   TIterator *fIterator; //collection iterator

protected:
   TIter() : fIterator(0) { }

public:
   TIter(const TCollection *col, Bool_t dir = kIterForward)
        : fIterator(col ? col->MakeIterator(dir) : 0) { }
   TIter(TIterator *it) : fIterator(it) { }
   TIter(const TIter &iter);
   TIter &operator=(const TIter &rhs);
   virtual ~TIter() { { if (fIterator) { delete fIterator; fIterator = 0; } } }
   TObject *operator()() { return fIterator ? fIterator->Next() : 0; }
   TObject *Next() { return fIterator ? fIterator->Next() : 0; }
   const TCollection *GetCollection() const { return fIterator ? fIterator->GetCollection() : 0; }
   Option_t *GetOption() const { return fIterator ? fIterator->GetOption() : ""; }
   void Reset() { if (fIterator) fIterator->Reset(); }

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 0; } static void Dictionary(); virtual TClass *IsA() const { return TIter::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TIter::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TCollection.h"; } static int DeclFileLine() { return 144; } static const char *ImplFileName(); static int ImplFileLine(); //Iterator wrapper
};


//---- ForEach macro -----------------------------------------------------------

// Macro to loop over all elements of a list of type "type" while executing
// procedure "proc" on each element
# 27 "/cdf/home/pcanal/scratch/code/root.merging/include/TSeqCollection.h" 2



class TSeqCollection : public TCollection {

protected:
   Bool_t fSorted; // true if collection has been sorted

   TSeqCollection() { }
   virtual void Changed() { fSorted = kFALSE; }

public:
   virtual ~TSeqCollection() { }
   virtual void Add(TObject *obj) { AddLast(obj); }
   virtual void AddFirst(TObject *obj) = 0;
   virtual void AddLast(TObject *obj) = 0;
   virtual void AddAt(TObject *obj, Int_t idx) = 0;
   virtual void AddAfter(TObject *after, TObject *obj) = 0;
   virtual void AddBefore(TObject *before, TObject *obj) = 0;
   virtual void RemoveFirst() { Remove(First()); }
   virtual void RemoveLast() { Remove(Last()); }
   virtual TObject *RemoveAt(Int_t idx) { return Remove(At(idx)); }
   virtual void RemoveAfter(TObject *after) { Remove(After(after)); }
   virtual void RemoveBefore(TObject *before) { Remove(Before(before)); }

   virtual TObject *At(Int_t idx) const = 0;
   virtual TObject *Before(TObject *obj) const = 0;
   virtual TObject *After(TObject *obj) const = 0;
   virtual TObject *First() const = 0;
   virtual TObject *Last() const = 0;
   Int_t LastIndex() const { return GetSize() - 1; }
   virtual Int_t IndexOf(const TObject *obj) const;
   virtual Bool_t IsSorted() const { return fSorted; }
   void UnSort() { fSorted = kFALSE; }

   static Int_t ObjCompare(TObject *a, TObject *b);
   static void QSort(TObject **a, Int_t first, Int_t last);
   static void QSort(TObject **a, TObject **b, Int_t first, Int_t last);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 0; } static void Dictionary(); virtual TClass *IsA() const { return TSeqCollection::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TSeqCollection::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TSeqCollection.h"; } static int DeclFileLine() { return 66; } static const char *ImplFileName(); static int ImplFileLine(); //Sequenceable collection ABC
};
# 27 "/cdf/home/pcanal/scratch/code/root.merging/include/TList.h" 2





const Bool_t kSortAscending = kTRUE;
const Bool_t kSortDescending = !kSortAscending;

class TObjLink;
class TListIter;


class TList : public TSeqCollection {

friend class TListIter;

protected:
   TObjLink *fFirst; //! pointer to first entry in linked list
   TObjLink *fLast; //! pointer to last entry in linked list
   TObjLink *fCache; //! cache to speedup sequential calling of Before() and After() functions
   Bool_t fAscending; //! sorting order (when calling Sort() or for TSortedList)

   TObjLink *LinkAt(Int_t idx) const;
   TObjLink *FindLink(const TObject *obj, Int_t &idx) const;
   TObjLink **DoSort(TObjLink **head, Int_t n);
   Bool_t LnkCompare(TObjLink *l1, TObjLink *l2);
   virtual TObjLink *NewLink(TObject *obj, TObjLink *prev = 0);
   virtual TObjLink *NewOptLink(TObject *obj, Option_t *opt, TObjLink *prev = 0);
   virtual void DeleteLink(TObjLink *lnk);

public:
   TList() { fFirst = fLast = fCache = 0; }
   TList(TObject *) { fFirst = fLast = fCache = 0; } // for backward compatibility, don't use
   virtual ~TList();
   virtual void Clear(Option_t *option="");
   virtual void Delete(Option_t *option="");
   virtual TObject *FindObject(const char *name) const;
   virtual TObject *FindObject(const TObject *obj) const;
   virtual TIterator *MakeIterator(Bool_t dir = kIterForward) const;

   virtual void Add(TObject *obj) { AddLast(obj); }
   virtual void Add(TObject *obj, Option_t *opt) { AddLast(obj, opt); }
   virtual void AddFirst(TObject *obj);
   virtual void AddFirst(TObject *obj, Option_t *opt);
   virtual void AddLast(TObject *obj);
   virtual void AddLast(TObject *obj, Option_t *opt);
   virtual void AddAt(TObject *obj, Int_t idx);
   virtual void AddAfter(TObject *after, TObject *obj);
   virtual void AddAfter(TObjLink *after, TObject *obj);
   virtual void AddBefore(TObject *before, TObject *obj);
   virtual void AddBefore(TObjLink *before, TObject *obj);
   virtual TObject *Remove(TObject *obj);
   virtual TObject *Remove(TObjLink *lnk);

   virtual TObject *At(Int_t idx) const;
   virtual TObject *After(TObject *obj) const;
   virtual TObject *Before(TObject *obj) const;
   virtual TObject *First() const;
   virtual TObjLink *FirstLink() const { return fFirst; }
   virtual TObject **GetObjectRef(TObject *obj) const;
   virtual TObject *Last() const;
   virtual TObjLink *LastLink() const { return fLast; }

   virtual void Sort(Bool_t order = kSortAscending);
   Bool_t IsAscending() { return fAscending; }

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 4; } static void Dictionary(); virtual TClass *IsA() const { return TList::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TList::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TList.h"; } static int DeclFileLine() { return 93; } static const char *ImplFileName(); static int ImplFileLine(); //Doubly linked list
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjLink                                                             //
//                                                                      //
// Wrapper around a TObject so it can be stored in a TList.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TObjLink {

friend class TList;

private:
   TObjLink *fNext;
   TObjLink *fPrev;
   TObject *fObject;

protected:
   TObjLink() { fNext = fPrev = this; fObject = 0; }

public:
   TObjLink(TObject *obj) : fNext(0), fPrev(0), fObject(obj) { }
   TObjLink(TObject *obj, TObjLink *lnk);
   virtual ~TObjLink() { }

   TObject *GetObject() const { return fObject; }
   TObject **GetObjectRef() { return &fObject; }
   void SetObject(TObject *obj) { fObject = obj; }
   virtual Option_t *GetAddOption() const { return ""; }
   virtual Option_t *GetOption() const { return fObject->GetOption(); }
   virtual void SetOption(Option_t *) { }
   TObjLink *Next() { return fNext; }
   TObjLink *Prev() { return fPrev; }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjOptLink                                                          //
//                                                                      //
// Wrapper around a TObject so it can be stored in a TList including    //
// an option string.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TObjOptLink : public TObjLink {

private:
   TString fOption;

public:
   TObjOptLink(TObject *obj, Option_t *opt) : TObjLink(obj), fOption(opt) { }
   TObjOptLink(TObject *obj, TObjLink *lnk, Option_t *opt) : TObjLink(obj, lnk), fOption(opt) { }
   ~TObjOptLink() { }
   Option_t *GetAddOption() const { return fOption.Data(); }
   Option_t *GetOption() const { return fOption.Data(); }
   void SetOption(Option_t *option) { fOption = option; }
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListIter                                                            //
//                                                                      //
// Iterator of linked list.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TListIter : public TIterator {

protected:
   const TList *fList; //list being iterated
   TObjLink *fCurCursor; //current position in list
   TObjLink *fCursor; //next position in list
   Bool_t fDirection; //iteration direction
   Bool_t fStarted; //iteration started

   TListIter() : fList(0), fCursor(0), fStarted(kFALSE) { }

public:
   TListIter(const TList *l, Bool_t dir = kIterForward);
   TListIter(const TListIter &iter);
   ~TListIter() { }
   TIterator &operator=(const TIterator &rhs);
   TListIter &operator=(const TListIter &rhs);

   const TCollection *GetCollection() const { return fList; }
   Option_t *GetOption() const;
   void SetOption(Option_t *option);
   TObject *Next();
   void Reset() { fStarted = kFALSE; }

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 0; } static void Dictionary(); virtual TClass *IsA() const { return TListIter::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TListIter::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TList.h"; } static int DeclFileLine() { return 186; } static const char *ImplFileName(); static int ImplFileLine(); //Linked list iterator
};
# 30 "/cdf/home/pcanal/scratch/code/root.merging/include/TNamed.h" 2






class TNamed : public TObject {

protected:
   TString fName; //object identifier
   TString fTitle; //object title

public:
   TNamed() { }
   TNamed(const char *name, const char *title) : fName(name), fTitle(title) { }
   TNamed(const TString &name, const TString &title) : fName(name), fTitle(title) { }
   TNamed(const TNamed &named);
   TNamed& operator=(const TNamed& rhs);
   virtual ~TNamed() { }
   virtual TObject *Clone(const char *newname="") const;
   virtual Int_t Compare(const TObject *obj) const;
   virtual void Copy(TObject &named);
   virtual void FillBuffer(char *&buffer);
   virtual const char *GetName() const {return fName.Data();}
   virtual const char *GetTitle() const {return fTitle.Data();}
   virtual ULong_t Hash() const { return fName.Hash(); }
   virtual Bool_t IsSortable() const { return kTRUE; }
   virtual void SetName(const char *name); // *MENU*
   virtual void SetNameTitle(const char *name, const char *title);
   virtual void SetTitle(const char *title=""); // *MENU*
   virtual void ls(Option_t *option="") const;
   virtual void Print(Option_t *option="") const;
   virtual Int_t Sizeof() const;

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 1; } static void Dictionary(); virtual TClass *IsA() const { return TNamed::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TNamed::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TNamed.h"; } static int DeclFileLine() { return 64; } static const char *ImplFileName(); static int ImplFileLine(); //The basis for a named object (name, title)
};
# 27 "/cdf/home/pcanal/scratch/code/root.merging/include/TShape.h" 2



# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TMaterial.h" 1
// @(#)root/g3d:$Name:  $:$Id: TMaterial.h,v 1.2 2000/12/13 15:13:46 brun Exp $
// Author: Rene Brun   03/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMaterial                                                            //
//                                                                      //
// Materials used in the Geometry Shapes                                //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
# 29 "/cdf/home/pcanal/scratch/code/root.merging/include/TMaterial.h"
# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TAttFill.h" 1
// @(#)root/base:$Name:  $:$Id: TAttFill.h,v 1.2 2000/12/13 15:13:45 brun Exp $
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttFill                                                             //
//                                                                      //
// Fill area attributes.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/Gtypes.h" 1
/* @(#)root/base:$Name:  $:$Id: Gtypes.h,v 1.6 2002/03/20 10:39:44 brun Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Gtypes                                                               //
//                                                                      //
// Types used by the graphics classes.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/Htypes.h" 1
/* @(#)root/base:$Name:  $:$Id: Htypes.h,v 1.3 2001/10/24 13:47:57 brun Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Htypes                                                               //
//                                                                      //
// Types used by the histogramming classes.                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////






typedef double Axis_t; //Axis values type (double)
typedef double Stat_t; //Statistics type (double)
# 25 "/cdf/home/pcanal/scratch/code/root.merging/include/Gtypes.h" 2


typedef short Font_t; //Font number (short)
typedef short Style_t; //Style number (short)
typedef short Marker_t; //Marker number (short)
typedef short Width_t; //Line width (short)
typedef short Color_t; //Color number (short)
typedef short SCoord_t; //Screen coordinates (short)
typedef double Coord_t; //Pad world coordinates (double)
typedef float Angle_t; //Graphics angle (float)
typedef float Size_t; //Attribute size (float)

enum EColor { kWhite, kBlack, kRed, kGreen, kBlue, kYellow, kMagenta, kCyan };
enum ELineStyle { kSolid = 1, kDashed, kDotted, kDashDotted };
enum EMarkerStyle {kDot=1, kPlus, kStar, kCircle=4, kMultiply=5,
                   kFullDotSmall=6, kFullDotMedium=7, kFullDotLarge=8,
                   kOpenTriangleDown = 16, kFullCross= 18,
                   kFullCircle=20, kFullSquare=21, kFullTriangleUp=22,
                   kFullTriangleDown=23, kOpenCircle=24, kOpenSquare=25,
                   kOpenTriangleUp=26, kOpenDiamond=27, kOpenCross=28,
                   kFullStar=29, kOpenStar=30};
# 26 "/cdf/home/pcanal/scratch/code/root.merging/include/TAttFill.h" 2



class TAttFill {

protected:
   Color_t fFillColor; //fill area color
   Style_t fFillStyle; //fill area style

public:
   TAttFill();
   TAttFill(Color_t fcolor,Style_t fstyle);
   virtual ~TAttFill();
   void Copy(TAttFill &attfill);
   Color_t GetFillColor() const { return fFillColor; }
   Style_t GetFillStyle() const { return fFillStyle; }
   Bool_t IsTransparent() const;
   virtual void Modify();
   virtual void ResetAttFill(Option_t *option="");
   virtual void SaveFillAttributes(ofstream &out, const char *name, Int_t coldef=1, Int_t stydef=1001);
   virtual void SetFillAttributes(); // *MENU*
   virtual void SetFillColor(Color_t fcolor) { fFillColor = fcolor; }
   virtual void SetFillStyle(Style_t fstyle) { fFillStyle = fstyle; }

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 1; } static void Dictionary(); virtual TClass *IsA() const { return TAttFill::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TAttFill::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TAttFill.h"; } static int DeclFileLine() { return 50; } static const char *ImplFileName(); static int ImplFileLine(); //Fill area attributes
};

inline Bool_t TAttFill::IsTransparent() const
{ return fFillStyle >= 4000 && fFillStyle <= 4100 ? kTRUE : kFALSE; }
# 30 "/cdf/home/pcanal/scratch/code/root.merging/include/TMaterial.h" 2


class TMaterial : public TNamed, public TAttFill {
 protected:
   Int_t fNumber; //Material matrix number
   Float_t fA; //A of Material
   Float_t fZ; //Z of Material
   Float_t fDensity; //Material density in gr/cm3
   Float_t fRadLength; //Material radiation length
   Float_t fInterLength; //Material interaction length

 public:
        TMaterial();
        TMaterial(const char *name, const char *title, Float_t a, Float_t z, Float_t density);
        TMaterial(const char *name, const char *title, Float_t a, Float_t z, Float_t density, Float_t radl, Float_t inter);
        virtual ~TMaterial();
        virtual Int_t GetNumber() const {return fNumber;}
        virtual Float_t GetA() const {return fA;}
        virtual Float_t GetZ() const {return fZ;}
        virtual Float_t GetDensity() const {return fDensity;}
        virtual Float_t GetRadLength() const {return fRadLength;}
        virtual Float_t GetInterLength() const {return fInterLength;}

        private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 3; } static void Dictionary(); virtual TClass *IsA() const { return TMaterial::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TMaterial::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TMaterial.h"; } static int DeclFileLine() { return 53; } static const char *ImplFileName(); static int ImplFileLine(); //Materials used in the Geometry Shapes
};
# 31 "/cdf/home/pcanal/scratch/code/root.merging/include/TShape.h" 2



# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TAttLine.h" 1
// @(#)root/base:$Name:  $:$Id: TAttLine.h,v 1.3 2000/12/13 15:13:45 brun Exp $
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttLine                                                             //
//                                                                      //
// Line attributes.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////






class TAttLine {

protected:
        Color_t fLineColor; //line color
        Style_t fLineStyle; //line style
        Width_t fLineWidth; //line width

public:
        TAttLine();
        TAttLine(Color_t lcolor,Style_t lstyle, Width_t lwidth);
        virtual ~TAttLine();
                void Copy(TAttLine &attline);
        Int_t DistancetoLine(Int_t px, Int_t py, Double_t xp1, Double_t yp1, Double_t xp2, Double_t yp2 );
        Color_t GetLineColor() const {return fLineColor;}
        Style_t GetLineStyle() const {return fLineStyle;}
        Width_t GetLineWidth() const {return fLineWidth;}
        virtual void Modify();
        virtual void ResetAttLine(Option_t *option="");
        virtual void SaveLineAttributes(ofstream &out, const char *name, Int_t coldef=1, Int_t stydef=1, Int_t widdef=1);
        virtual void SetLineAttributes(); // *MENU*
        virtual void SetLineColor(Color_t lcolor) { fLineColor = lcolor;}
        virtual void SetLineStyle(Style_t lstyle) { fLineStyle = lstyle;}
        virtual void SetLineWidth(Width_t lwidth) { fLineWidth = lwidth;}

        private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 1; } static void Dictionary(); virtual TClass *IsA() const { return TAttLine::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TAttLine::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TAttLine.h"; } static int DeclFileLine() { return 53; } static const char *ImplFileName(); static int ImplFileLine(); //Line attributes
};
# 35 "/cdf/home/pcanal/scratch/code/root.merging/include/TShape.h" 2







# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TAtt3D.h" 1
// @(#)root/base:$Name:  $:$Id: TAtt3D.h,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Fons Rademakers   08/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAtt3D                                                               //
//                                                                      //
// Use this attribute class when an object should have 3D capabilities. //
//                                                                      //
//////////////////////////////////////////////////////////////////////////






class TAtt3D {

public:
   TAtt3D() { }
   virtual ~TAtt3D() { }

   virtual void Sizeof3D() const;

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 1; } static void Dictionary(); virtual TClass *IsA() const { return TAtt3D::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TAtt3D::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TAtt3D.h"; } static int DeclFileLine() { return 37; } static const char *ImplFileName(); static int ImplFileLine(); //3D attributes
};
# 43 "/cdf/home/pcanal/scratch/code/root.merging/include/TShape.h" 2



# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/X3DBuffer.h" 1
/* @(#)root/g3d:$Name:  $:$Id: X3DBuffer.h,v 1.1.1.1 2000/05/16 17:00:43 rdm Exp $ */
/* Author: Nenad Buncic   13/12/95*/

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/




# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/DllImport.h" 1
# 16 "/cdf/home/pcanal/scratch/code/root.merging/include/X3DBuffer.h" 2

typedef struct _x3d_data_ {
      int numPoints;
      int numSegs;
      int numPolys;
    float *points; /* x0, y0, z0, x1, y1, z1, ..... ..... ....    */
      int *segs; /* c0, p0, q0, c1, p1, q1, ..... ..... ....    */
      int *polys; /* c0, n0, s0, s1, ... sn, c1, n1, s0, ... sn  */
} X3DBuffer;


typedef struct _x3d_sizeof_ {
      int numPoints;
      int numSegs;
      int numPolys;
} Size3D;


extern "C" int AllocateX3DBuffer ();
extern "C" void FillX3DBuffer (X3DBuffer *buff);





R__EXTERN Size3D gSize3D;
# 47 "/cdf/home/pcanal/scratch/code/root.merging/include/TShape.h" 2



# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TPolyLine3D.h" 1
// @(#)root/g3d:$Name:  $:$Id: TPolyLine3D.h,v 1.5 2002/01/20 10:02:40 brun Exp $
// Author: Nenad Buncic   17/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/





//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPolyLine3D                                                          //
//                                                                      //
// A 3-D polyline.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
# 31 "/cdf/home/pcanal/scratch/code/root.merging/include/TPolyLine3D.h"
# 1 "/cdf/home/pcanal/scratch/code/root.merging/include/TString.h" 1
# 32 "/cdf/home/pcanal/scratch/code/root.merging/include/TPolyLine3D.h" 2
# 44 "/cdf/home/pcanal/scratch/code/root.merging/include/TPolyLine3D.h"
class TPolyLine3D : public TObject, public TAttLine, public TAtt3D {

protected:
   Int_t fN; //Number of points
   Float_t *fP; //[3*fN] Array of 3-D coordinates  (x,y,z)
   TString fOption; //options
   UInt_t fGLList; //!The list number for OpenGL view
   Int_t fLastPoint; //The index of the last filled point

public:
   TPolyLine3D();
   TPolyLine3D(Int_t n, Option_t *option="");
   TPolyLine3D(Int_t n, Float_t *p, Option_t *option="");
   TPolyLine3D(Int_t n, Double_t *p, Option_t *option="");
   TPolyLine3D(Int_t n, Float_t *x, Float_t *y, Float_t *z, Option_t *option="");
   TPolyLine3D(Int_t n, Double_t *x, Double_t *y, Double_t *z, Option_t *option="");
   TPolyLine3D(const TPolyLine3D &polylin);
   virtual ~TPolyLine3D();

   virtual void Copy(TObject &polyline);
   virtual Int_t DistancetoPrimitive(Int_t px, Int_t py);
   virtual void Draw(Option_t *option="");
   virtual void DrawPolyLine(Int_t n, Float_t *p, Option_t *option="");
   virtual void ExecuteEvent(Int_t event, Int_t px, Int_t py);
   Int_t GetLastPoint() const {return fLastPoint;}
   Int_t GetN() const {return fN;}
   Float_t *GetP() const {return fP;}
   Option_t *GetOption() const {return fOption.Data();}
   virtual void ls(Option_t *option="") const;
   virtual Int_t Merge(TCollection *list);
   virtual void Paint(Option_t *option="");
   virtual void PaintPolyLine(Int_t n, Float_t *p, Option_t *option="");
   virtual void PaintPolyLine(Int_t n, Double_t *p, Option_t *option="");
   virtual void Print(Option_t *option="") const;
   virtual void SavePrimitive(ofstream &out, Option_t *option);
   virtual Int_t SetNextPoint(Double_t x, Double_t y, Double_t z); // *MENU*
   virtual void SetOption(Option_t *option="") {fOption = option;}
   virtual void SetPoint(Int_t point, Double_t x, Double_t y, Double_t z); // *MENU*
   virtual void SetPolyLine(Int_t n, Option_t *option="");
   virtual void SetPolyLine(Int_t n, Float_t *p, Option_t *option="");
   virtual void SetPolyLine(Int_t n, Double_t *p, Option_t *option="");
   virtual void Sizeof3D() const;
   virtual Int_t Size() const { return fLastPoint+1;}

   static void DrawOutlineCube(TList *outline, Double_t *rmin, Double_t *rmax);

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 1; } static void Dictionary(); virtual TClass *IsA() const { return TPolyLine3D::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TPolyLine3D::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TPolyLine3D.h"; } static int DeclFileLine() { return 90; } static const char *ImplFileName(); static int ImplFileLine(); //A 3-D polyline
};
# 51 "/cdf/home/pcanal/scratch/code/root.merging/include/TShape.h" 2


class TNode;

class TShape : public TNamed, public TAttLine, public TAttFill, public TAtt3D {

protected:
   Int_t fNumber; //Shape number
   Int_t fVisibility; //Visibility flag
   TMaterial *fMaterial; //Pointer to material

   Int_t ShapeDistancetoPrimitive(Int_t numPoints, Int_t px, Int_t py);

public:
                   TShape();
                   TShape(const char *name, const char *title, const char *material);
   virtual ~TShape();
   TMaterial *GetMaterial() const {return fMaterial;}
   virtual Int_t GetNumber() const {return fNumber;}
           Int_t GetVisibility() const {return fVisibility;}
   virtual void Paint(Option_t *option="");
   virtual void PaintGLPoints(Float_t *vertex);
   virtual void PaintShape(X3DBuffer *buff, Bool_t rangeView=kFALSE);
   virtual void SetName(const char *name);
   virtual void SetPoints(Float_t *buffer);
   virtual void SetVisibility(Int_t vis) {fVisibility = vis;} // *MENU*

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 2; } static void Dictionary(); virtual TClass *IsA() const { return TShape::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { TShape::Streamer(b); } static const char *DeclFileName() { return "/cdf/home/pcanal/scratch/code/root.merging/include/TShape.h"; } static int DeclFileLine() { return 78; } static const char *ImplFileName(); static int ImplFileLine(); //Basic shape
};

R__EXTERN TNode *gNode;

inline void TShape::PaintGLPoints(Float_t *) { }
inline void TShape::SetName(const char *) { }
# 8 "Simple.h" 2
# 1 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h" 1
/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * I/O stream header file iostream.h
 ************************************************************************
 * Description:
 *  CINT iostream header file
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/



/*********************************************************************
* Try initializaing precompiled iostream library
*********************************************************************/
# 35 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h"
//typedef strstream stringstream;
# 45 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h"
/*********************************************************************
* Use fake iostream only if precompiled version does not exist.
*********************************************************************/







/*********************************************************************
* ios
*
*********************************************************************/


//class io_state;
# 125 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h"
/*********************************************************************
* ostream
*
*********************************************************************/

                /* virtual */
# 246 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h"
/* instanciation of cout,cerr */




/*********************************************************************
* istream
*
*********************************************************************/

                /* virtual */
# 393 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h"
/* instanciation of cin */


/*********************************************************************
* iostream
*
*********************************************************************/







/*********************************************************************
* ofstream, ifstream 
*
*********************************************************************/
# 452 "/cdf/home/pcanal/scratch/code/root.merging/cint/include/iostream.h"
/*********************************************************************
* iostream manipurator emulation
*
*  Following description must be deleted when pointer to compiled 
* function is fully supported.
*********************************************************************/
# 9 "Simple.h" 2

class Simple : public TObject {

private:
   Int_t fID; // id number
   TShape* fShape; // pointer to base class shape

public:

   Simple() : fID(0), fShape(0) { }
   Simple(Int_t id, TShape* shape): fID(id), fShape(shape) { }
   virtual ~Simple();
   virtual void Print(Option_t *option = "") const;

   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 1; } static void Dictionary(); virtual TClass *IsA() const { return Simple::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { Simple::Streamer(b); } static const char *DeclFileName() { return "Simple.h"; } static int DeclFileLine() { return 23; } static const char *ImplFileName(); static int ImplFileLine(); //Simple class
};
# 6 "Simple.cxx" 2

static int R__dummyint 7 = ROOT::ClassInfo<Simple >::SetImplFile("Simple.cxx", 7);

Simple::~Simple() {
// Destructor
  if (fShape) {
    delete fShape;
    fShape =0;
  }
}

void Simple::Print(Option_t *option) const {
  // Print the contents
  cout << "fID= " << fID << endl;
  fShape -> Print();

}
