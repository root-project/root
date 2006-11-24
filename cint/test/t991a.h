/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef CLASS_HH_
#define CLASS_HH_

//#include <TObject.h>
#include "t991.h"
#include <stdio.h>
#include <typeinfo>

namespace Master
{  
  class Container : public TObject{
  public:
    Container() 
    {}

    template<class T> T & func(T * t  ) {
      printf("Master::Container::func(%s)\n",typeid(T).name());
      return *t; 
    }
    void Print(Option_t *) { }
  protected:
    //ClassDef(Master::Container,1);
  };

  class Object : public TObject {
  public:
    Object() {}
    
  protected:
    //ClassDef(Master::Object,1);
  };
}

#ifdef __MAKECINT__
#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;
#endif


#endif

