#ifndef RecRecordImp_cxx
#define RecRecordImp_cxx

#include <iostream>

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

  std::cout << "RecRecordImp::Print Header:" << std::endl;
  fHeader.Print();

  return;

}

#endif



