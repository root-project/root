/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef STUB_H
#define STUB_H

#include "Complex.h"

#ifdef __CINT__

void Display(Complex a);

#else

void Display();

#endif


#endif
