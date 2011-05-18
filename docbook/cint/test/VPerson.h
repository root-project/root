/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef VPERSON_H
#define VPERSON_H

#include "VString.h"
#include "VObject.h"

class VPerson : public VObject
{
public:
   VPerson();
   VPerson(const VPerson& x);
   VPerson(const char* nameIn, const char* syozokuIn);
   VPerson(const char* nameIn, int num);
   VPerson& operator=(const VPerson& x);
   ~VPerson();
   void set(const char* nameIn, const char* shozokuIn);

   const char* Name()
   {
      return name.String();
   }

   const char* Syozoku()
   {
      return syozoku.String();
   }

   void disp();

private:
   VString name;
   VString syozoku;
};

#endif // VPERSON_H
