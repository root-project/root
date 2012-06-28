/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t992.h"
#endif

#include <stdio.h>

int main() {
  vector<vector<Int_t> > vvi;
  vector<Int_t> vi;
  int i,j;
  for(i=0;i<4;i++) {
    for(j=0;j<i+1;j++) {
      vi.push_back(i*100+j);
    }
    vvi.push_back(vi);
    vi.erase(vi.begin(),vi.end());
  }

  vector<vector<Int_t> >::iterator first1=vvi.begin();
  vector<vector<Int_t> >::iterator last1 =vvi.end();
  while(first1!=last1) {
    vector<Int_t>::iterator first2=(*first1).begin();
    vector<Int_t>::iterator last2 =(*first1).end();
    while(first2!=last2) {
      printf("%d ",*first2);
      ++first2;
    }
      printf("\n");
    ++first1;
  }

  return 0;
}
