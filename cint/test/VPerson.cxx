/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "VPerson.h"

#include <iostream>

using namespace std;

VPerson::VPerson()
: VObject()
, name()
, syozoku()
{
}

VPerson::VPerson(const VPerson& x)
: VObject(x)
, name(x.name)
, syozoku(x.syozoku)
{
}

VPerson::VPerson(const char* nameIn, const char* syozokuIn)
: VObject()
, name(nameIn)
, syozoku(syozokuIn)
{
}

VPerson::VPerson(const char* nameIn, int num)
: VObject()
, name(nameIn)
{
   char buf[256];
   sprintf(buf, "syozoku%d", num);
   syozoku = buf;
}

VPerson& VPerson::operator=(const VPerson& x)
{
   if (this != &x) {
      name = x.name;
      syozoku = x.syozoku;
   }
   return *this;
}

VPerson::~VPerson()
{
}

void VPerson::set(const char* nameIn, const char* syozokuIn)
{
   name = nameIn;
   syozoku = syozokuIn;
}

void VPerson::disp()
{
   cout << name.String() << " " << syozoku.String() << endl;
}

