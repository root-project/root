#ifndef RecRecordImp_cxx
#ifdef ClingWorkAroundMultipleInclude
#define RecRecordImp_cxx
#endif

#include <iostream>
using namespace std;

#include "RecRecordImp.h"




template<class T>
void  RecRecordImp<T>::Print(Option_t* /* option */) const {
  //
  //  Purpose:  Print header in form supported by TObject::Print.
  //
  //  Arguments: option (not used)
  //
  //  Return:  none.
  //
  //  Contact:   S. Kasahara
  // 

  cout << "RecRecordImp::Print Header:" << endl;
  fHeader.Print();

  return;

}

#endif



