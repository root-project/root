/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "array.h"
#include "matrix.h"

class matrix : public Matrix {
  int m_ptr;
 public:
  matrix(const string& fmtin="%s ") : Matrix(fmtin) { m_ptr=0; }

  matrix& operator>>(array& a) {
    int n=size();
    a.resize(n);
    for(int i=0;i<n;i++) a[i] = atof(get(m_ptr,i).c_str());
    ++m_ptr;
    return(*this);
  }
  matrix& operator>>(carray& a) {
    int n=size();
    a.resize(n);
    for(int i=0;i<n;i++) {
      a.re[i] = atof(get(m_ptr,i).c_str());
      a.im[i] = atof(get(m_ptr+1,i).c_str());
    }
    m_ptr+=2;
    return(*this);
  }

  matrix& operator<<(array& a) {
    int n=a.size();
    for(int i=0;i<n;i++) {
      if(size()<i) push_back(Line());
    }
    // TODO
    return(*this);
  }
  matrix& operator<<(carray& a) {
    // TODO
    return(*this);
  }

  matrix& operator<<(G__CINT_ENDL c) { m_ptr=0; return(*this); }
  matrix& operator>>(G__CINT_ENDL c) { m_ptr=0; return(*this); }

};
