/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "VCompany.h"

#include <iostream>

using namespace std;

VCompany::VCompany()
      : VObject()
{
}

VCompany::VCompany(const VCompany& x)
: VObject()
, name(x.name)
, country(x.country)
{
}

VCompany::VCompany(const char* nameIn, const char* countryIn)
: VObject()
, name(nameIn)
, country(countryIn)
{
}

VCompany::VCompany(const char* nameIn, int num)
: VObject()
, name(nameIn)
{
   char buf[256];
   sprintf(buf, "country%d", num);
   country = buf;
}

VCompany& VCompany::operator=(const VCompany& x)
{
   if (this != &x) {
      name = x.name;
      country = x.country;
   }
   return *this;
}

VCompany::~VCompany()
{
}

void VCompany::set(const char* nameIn, const char* countryIn)
{
   name = nameIn;
   country = countryIn;
}

void VCompany::disp()
{
   cout << name.String() << " " << country.String() << endl;
}

