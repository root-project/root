/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* matrix.h
*
* Matrix class , I/F to Excel csv file
*
**************************************************************************/

#if defined(__CINT__) && !defined(__MAKECINT__)
#pragma include_noerr "matrix.dll"
#endif

#ifndef G__MATRIX_H
#define G__MATRIX_H

#include <vector>
#include <string>
#include <cstdio>
#include "ReadF.h"
using namespace std;

///////////////////////////////////////////////////////////////////////////
enum column {
 A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,
 AA,AB,AC,AD,AE,AF,AG,AH,AI,AJ,AK,AL,AM,AN,AO,AP,AQ,AR,AS,AT,AU,AV,AW,AX,AY,AZ,
 BA,BB,BC,BD,BE,BF,BG,BH,BI,BJ,BK,BL,BM,BN,BO,BP,BQ,BR,BS,BT,BU,BV,BW,BX,BY,BZ,
 CA,CB,CC,CD,CE,CF,CG,CH,CI,CJ,CK,CL,CM,CN,CO,CP,CQ,CR,CS,CT,CU,CV,CW,CX,CY,CZ,
 DA,DB,DC,DD,DE,DF,DG,DH,DI,DJ,DK,DL,DM,DN,DO,DP,DQ,DR,DS,DT,DU,DV,DW,DX,DY,DZ,
};

///////////////////////////////////////////////////////////////////////////
typedef vector<string> Line;

///////////////////////////////////////////////////////////////////////////
class Matrix : public vector<Line> {
  string fmt;
 public:
  Matrix(const string& fmtin="%s ") : fmt(fmtin) { }
  void setfmt(const string& fmtin) { fmt = fmtin; }
  typedef vector<Line>::iterator iterator;

  Matrix range(unsigned  int x1,unsigned int x2
	       ,unsigned int y1,unsigned int y2) ;

  int readcsv(const string& fname) ;

  string& operator()(unsigned int x,unsigned int y) ;
  string& get(unsigned int x,unsigned int y) { return(operator()(x,y)); }

  void disp(int y1i=0,unsigned int sz=1000) const ;
};


#if defined(__CINT__) && !defined(__MAKECINT__)
#include "matrix.cxx"
#endif

#endif


