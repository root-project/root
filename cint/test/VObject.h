/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef VOBJECT_H
#define VOBJECT_H

#include "VType.h"

#include <iostream>

class VObject
{
public:
   VObject();
   VObject(const VObject& x);
   virtual VObject& operator=(const VObject& x);
   virtual ~VObject();
   virtual void disp()
   {
      std::cout << "(VObject)" << std::endl;
   }
};

#endif // VOBJECT_H
