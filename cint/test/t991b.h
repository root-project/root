/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef SLAVE1_HH_
#define SLAVE1_HH_

//#include <TObject.h>
#include "t991.h"


namespace Slave1
{  

  class Object : public TObject {
  public:
    Object() {}
    
  protected:
    //ClassDef(Slave1::Object,1);
  };
}

#include "t991a.h"

#ifdef __MAKECINT__
#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;
#endif

#endif
