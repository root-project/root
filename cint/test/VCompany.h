/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef VCOMPANY_H
#define VCOMPANY_H

#include "VObject.h"
#include "VString.h"

class VCompany : public VObject
{
public:
   VCompany();
   VCompany(const VCompany& x);
   VCompany(const char* nameIn, const char* countryIn);
   VCompany(const char* nameIn, int num);
   VCompany& operator=(const VCompany& x);
   ~VCompany();
   void set(const char* nameIn, const char* countryIn);

   const char* Name()
   {
      return name.String();
   }

   const char* Syozoku()
   {
      return country.String();
   }

   void disp();

private:
   VString name;
   VString country;
};

#endif // VCOMPANY_H
