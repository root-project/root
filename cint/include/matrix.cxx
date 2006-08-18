/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "matrix.h"

//////////////////////////////////////////////////////////////////////
// class Matrix
//////////////////////////////////////////////////////////////////////
Matrix Matrix::range(unsigned  int x1,unsigned int x2
		     ,unsigned int y1,unsigned int y2) {
  Matrix mat2(fmt);
  Line line;
  for(unsigned int y=y1;y<=y2;++y) {
    line.clear();
    if(size()>y) {
      for(unsigned int x=x1;x<=x2;++x) {
	if((*this)[y].size()>x) line.push_back((*this)[y][x]);
	else                    line.push_back("");
      }
    }
    mat2.push_back(line);
  }
  return(mat2);
}

//////////////////////////////////////////////////////////////////////
int Matrix::readcsv(const string& fname) {
  ReadFile f(fname.c_str());
  if(!f.isvalid()) {
    fprintf(stderr,"ReadFile: Can not open %s\n",fname.c_str());
    return(0);
  }
  Line line;
  f.setdelimitor(",");
  f.setseparator("");
  while(f.read()) {
    //printf("%d\n",f.argc);
    line.clear();
    for(int i=1;i<=f.argc;i++) line.push_back(f.argv[i]);
    push_back(line);
  }
  return(1);
}

//////////////////////////////////////////////////////////////////////
string& Matrix::operator()(unsigned int x,unsigned int y) {
  static string strnull;
  if(size()<=y) return(strnull);
  Line *line = &(*this)[y];
  if(line->size()<=x) return(strnull);
  return(line->operator[](x));
}

//////////////////////////////////////////////////////////////////////
void Matrix::disp(int y1i,unsigned int sz) const {
  char num[20];
  unsigned int y1;
  if(y1i>=0) y1=y1i;
  else       y1=size()+y1i;
  for(unsigned int y=y1;y<=y1+sz;++y) {
    if(size()>y) {
      const Line *line = &operator[](y);
      if(y==y1) {
	int x=0;
	printf("%5s:","");
	for(Line::const_iterator j=line->begin();j!=line->end();++j) {
	  sprintf(num,"%d",x++);
	  printf(fmt.c_str(),num);
	}
	printf("\n");
      }
      printf("%5d:",y);
      for(Line::const_iterator j=line->begin();j!=line->end();++j) {
	printf(fmt.c_str(),(*j).c_str());
      }
      printf("\n");
    }
  }
}

