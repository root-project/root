/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "VObject.h"

#include <iostream>

using namespace std;

VObject::VObject()
{
}

VObject::VObject(const VObject& x)
{
}

VObject& VObject::operator=(const VObject& x)
{
   cerr << "VObject::operator=() must be overridden" << endl;
   return *this;
}

VObject::~VObject()
{
}

