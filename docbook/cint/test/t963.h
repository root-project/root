/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// resolved
// 021211rootcint.txt, Maarten 
// 021214rootcint.txt
// masa/*
//  forward class declaration and enclosed class definition causes 
//  member function to be mis-interpreted.

#include <stdio.h>

template <typename T> class TPhTopoData;
typedef TPhTopoData<int> TPhTopoDataI;

template <typename T> class TPhTopoData {
 public:
  TPhTopoData() {}
  TPhTopoData(const TPhTopoData &a);
  void disp() const { printf("TPhTopoData<T>.disp()\n"); }
  class TError  {
   public:
    void disp() const { printf("TPhTopoData<T>::TError.disp()\n"); }
  };
};

template <typename T> TPhTopoData<T>::TPhTopoData(const TPhTopoData<T> &a) {
  printf("TPhTopoData<T>::TPhTopoData()\n");
}

// tmplt.c line 2316  ON1587
