/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif
#include <stdio.h>
#ifdef DLL
#include "vec3d.dll"
#else
#include "vec3d.h"
#endif

// Please use following notation for now
// This is more efficient.
test1() {
  int i,j,k;
  const int x=5, y=3,z=4;
  float3d a(x,y,z);
  for(i=0;i<x;i++) {
    for(j=0;j<y;j++) {
      for(k=0;k<z;k++) {
        a(i,j,k) = i+j+k;
      }
    }
  }

  for(i=0;i<x;i++) {
    for(j=0;j<y;j++) {
      for(k=0;k<z;k++) {
        printf("i=%d j=%d k=%d %g\n",i,j,k,a(i,j,k));
      }
    }
  }
}

// Please don't use it as follows
// This is slightly inefficient and cint has bug now
test2() {
  int i,j,k;
  const int x=5, y=3,z=4;
  float3d a(x,y,z);
  for(i=0;i<x;i++) {
    for(j=0;j<y;j++) {
      for(k=0;k<z;k++) {
        a[i][j][k] = i+j+k;
      }
    }
  }

  for(i=0;i<x;i++) {
    for(j=0;j<y;j++) {
      for(k=0;k<z;k++) {
        printf("i=%d j=%d k=%d %g\n",i,j,k,a[i][j][k]);
	cout << a[i][j][k] << endl;	
      }
    }
  }
}

main() {
  test1();
  test2();
}
