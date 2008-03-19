/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***********************************************************************
* VObject.h , C++
*
************************************************************************
* Description:
*
***********************************************************************/

#ifndef VOBJECT_H
#define VOBJECT_H

#include "VType.h"

class VObject {
 public:
  VObject();
  VObject(VObject& x);
  virtual VObject& operator=(VObject& x);
  virtual ~VObject();
  virtual void disp() { 
    cout << "(VObject)" << endl;  
  }
};

#endif

