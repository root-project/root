#include "header.h"
#include "Rtypes.h"

#ifndef __CINT__
RootClassVersion(Hard2Stream,2);
#endif

#ifdef __CINT__
#pragma link C++ class Hard2Stream-;
#endif

#include <iostream>

void Hard2Stream::print() {
   std::cout << "Hard2Stream: " /* << (void*)this */ << std::endl;
   std::cout << "val : " << getVal() << std::endl;
}


RootStreamer(Hard2Stream,hard2StreamStreamer);
