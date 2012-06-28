/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

class arr2 {
 public:
  double *p;
  double& operator[](int n) { return p[n]; }
};


class arr {
 public:
  double dat[10][10];
  arr2 operator[](int n) { 
    arr2 t;
    t.p = dat[n];
    return(t);
  }
};

