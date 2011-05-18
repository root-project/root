/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/iter.h

#include <iterator>
using namespace std;

#ifndef G__ITERATOR_DLL
#define G__ITERATOR_DLL
#endif

typedef input_iterator<int,long> input_iteratorX;
typedef forward_iterator<int,long> forward_iteratorX;
typedef bidirectional_iterator<int,long> bidirectional_iteratorX;
typedef random_access_iterator<int,long> random_access_iteratorX;

#if 1
input_iterator_tag iterator_category(const input_iterator<int,long>& x) {
  return input_iterator_tag();
}

output_iterator_tag iterator_category(const output_iterator& x) {
  return output_iterator_tag();
}

forward_iterator_tag iterator_category(const forward_iterator<int,long>& x) {
  return forward_iterator_tag();
}

bidirectional_iterator_tag 
iterator_category(const bidirectional_iterator<int,long>& x) {
  return bidirectional_iterator_tag();
}

random_access_iterator_tag
iterator_category(const random_access_iterator<int,long>& x) {
  return random_access_iterator_tag();
}
#endif

#ifdef __MAKECINT__
#pragma link C++ global G__ITERATOR_DLL;
#pragma link C++ all functions;
#pragma link C++ all classes;
#endif

