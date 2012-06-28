/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifndef SLAVE2_HH_
#define SLAVE2_HH_

//#include <TObject.h>
#include "t991.h"
#include "t991a.h"


namespace Slave2
{  

  class Object : public TObject {
  public:
    Object() {}
    
  protected:
    //ClassDef(Slave2::Object,1);
  };
}

#ifdef __MAKECINT__
#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;
//#pragma link C++ class Master::Container;
#pragma link C++ namespace Slave2;
#pragma link C++ class Slave2::Object;
#pragma link C++ function Master::Container::func(Slave2::Object *);
#endif


#endif

