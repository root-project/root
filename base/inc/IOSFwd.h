// @(#)root/base:$Name:$:$Id:$
// Author: Fons Rademakers   23/1/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_IOSFwd
#define ROOT_IOSFwd

#ifndef ROOT_RConfig
#include "RConfig.h"
#endif

#if defined(R__ANSISTREAM)
#   if defined(R__TMPLTSTREAM)
#      include <iostream>
#   else
#      include <iosfwd>
#   endif
using namespace std;
#elif R__MWERKS
template <class charT> class ios_traits;
template <class charT, class traits> class basic_istream;
template <class charT, class traits> class basic_ostream;
template <class charT, class traits> class basic_fstream;
template <class charT, class traits> class basic_ofstream;
template <class charT, class traits> class basic_ifstream;
typedef basic_istream<char, ios_traits<char> > istream;
typedef basic_ostream<char, ios_traits<char> > ostream;
typedef basic_fstream<char, ios_traits<char> > fstream;
typedef basic_ofstream<char, ios_traits<char> > ofstream;
typedef basic_ifstream<char, ios_traits<char> > ifstream;
#else
class istream;
class ostream;
class fstream;
class ifstream;
class ofstream;
#endif

#endif
