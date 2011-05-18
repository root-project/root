/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef VTYPE_H
#define VTYPE_H

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define SUCCESS 0
#define FAILURE 1

/***********************************************************************
* C++ の基本型は必ず typedef で別名をつけて使用すること。
*
***********************************************************************/
typedef unsigned char Bool_t;
typedef int Int_t;
typedef short Short_t;
typedef unsigned int Uint_t;
typedef unsigned int Size_t;
typedef unsigned long Flags_t;
typedef long PVoid_t;
typedef double Double_t;
typedef float Float_t;
typedef char Char_t;

#define MIN(a,b) (a>b?b:a)
#define MAX(a,b) (a>b?a:b)

#define MATCH 1
#define UNMATCH 0

#endif // VTYPE_H
