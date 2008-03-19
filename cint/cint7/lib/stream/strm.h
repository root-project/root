/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file stream.sut.h
 ************************************************************************
 * Description:
 *  Stub file for making iostream library
 ************************************************************************
 * Copyright(c) 1991~2001,  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__STREAM_H
#define G__STREAM_H

#ifndef __CINT__

#ifndef G__NEWSTDHEADER
#if defined(__APPLE__)
#include <iostream.h>
#else
#include <stream.h>
#endif
#endif

#else

#include "iostrm.h"
#include "stdiostr.h"
#include "fstrm.h"
/* for filebuf */

#ifdef G__NEVER
extern char*  oct(long, int =0);
extern char*  dec(long, int =0);
extern char*  hex(long, int =0);
extern char*  chr(int, int =0);	/* chr(0) is the empty string "" */
extern char*  str(const char*, int =0);
extern char*  form(const char* ...);
#endif

/* printf format
 * Things may go terribly wrong (maybe even core
 * dumps, if form tries to create a string with
 * more than "max_field_width" characters. */

/* WS used to be a special in streams. The WS manipulator
 * is implemented differently but may be extracted from an istream
 * with the same effect as the old form.
 */

// extern istream& WS(istream&); // alpha cxx didn't like this
// extern void eatwhite(istream&);     

static const int input = (ios::in) ;
static const int output = (ios::out) ;
static const int append = (ios::app) ;
static const int atend = (ios::ate) ;
static const int _good = (ios::goodbit) ;
static const int _bad = (ios::badbit) ;
static const int _fail = (ios::failbit) ;
static const int _eof = (ios::eofbit) ;

typedef ios::io_state state_value ;

#endif
#endif
