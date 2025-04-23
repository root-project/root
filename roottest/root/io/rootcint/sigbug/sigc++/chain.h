// -*- c++ -*-
//   Copyright 2001, Karl Einar Nelson
/* This is a generated file, do not edit.  Generated from template.macros.m4 */

#ifndef   SIGC_CHAIN_H
#define   SIGC_CHAIN_H
#include <sigc++/slot.h>
#include <sigc++/bind.h>

// FIXME this is a quick hack - needs to become proper adaptor class
//  with handling of notify from both slots.  Needs stuff for void 
//  return broken compilers

/*
  SigC::chain
  -------------
  chain() binds two slots as a unified call.  Chain takes two
  slots and returns a third.  The second slot is called the
  getter.  It will receive all of the parameters of the resulting
  slot.  The first slot, the setter, will receive the return value 
  from the getter and its return will be the return value of the 
  combined slot.  An arbitrary number of chains can be set up taking 
  the return of one slot and passing it to the parameters of the next.

  Simple Sample usage:

    float get(int i)  {return i+1;}
    double set(float) {return i+2;}

    Slot1<double,int>   s1=chain(slot(&set),slot(&get)); 
    s1(1);  // set(get(1));

*/

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif





template <class R,class C> 
R chain0_(Slot1<R,C> setter,Slot0<C> getter)
  { return setter(getter()); } 
template <class R,class C> 
Slot0<R> chain(const Slot1<R,C>& setter,const Slot0<C>& getter) 
  { 
    typedef R (*Func)(Slot1<R,C>,Slot0<C>);
    Func func=&chain0_<R,C>;
    return bind(slot(func),setter,getter);
  }


template <class R,class C,class P1> 
R chain1_(P1 p1,Slot1<R,C> setter,Slot1<C,P1> getter)
  { return setter(getter(p1)); } 
template <class R,class C,class P1> 
Slot1<R,P1> chain(const Slot1<R,C>& setter,const Slot1<C,P1>& getter) 
  { 
    typedef R (*Func)(P1,Slot1<R,C>,Slot1<C,P1>);
    Func func=&chain1_<R,C,P1>;
    return bind(slot(func),setter,getter);
  }


template <class R,class C,class P1,class P2> 
R chain2_(P1 p1,P2 p2,Slot1<R,C> setter,Slot2<C,P1,P2> getter)
  { return setter(getter(p1,p2)); } 
template <class R,class C,class P1,class P2> 
Slot2<R,P1,P2> chain(const Slot1<R,C>& setter,const Slot2<C,P1,P2>& getter) 
  { 
    typedef R (*Func)(P1,P2,Slot1<R,C>,Slot2<C,P1,P2>);
    Func func=&chain2_<R,C,P1,P2>;
    return bind(slot(func),setter,getter);
  }


template <class R,class C,class P1,class P2,class P3> 
R chain3_(P1 p1,P2 p2,P3 p3,Slot1<R,C> setter,Slot3<C,P1,P2,P3> getter)
  { return setter(getter(p1,p2,p3)); } 
template <class R,class C,class P1,class P2,class P3> 
Slot3<R,P1,P2,P3> chain(const Slot1<R,C>& setter,const Slot3<C,P1,P2,P3>& getter) 
  { 
    typedef R (*Func)(P1,P2,P3,Slot1<R,C>,Slot3<C,P1,P2,P3>);
    Func func=&chain3_<R,C,P1,P2,P3>;
    return bind(slot(func),setter,getter);
  }


template <class R,class C,class P1,class P2,class P3,class P4> 
R chain4_(P1 p1,P2 p2,P3 p3,P4 p4,Slot1<R,C> setter,Slot4<C,P1,P2,P3,P4> getter)
  { return setter(getter(p1,p2,p3,p4)); } 
template <class R,class C,class P1,class P2,class P3,class P4> 
Slot4<R,P1,P2,P3,P4> chain(const Slot1<R,C>& setter,const Slot4<C,P1,P2,P3,P4>& getter) 
  { 
    typedef R (*Func)(P1,P2,P3,P4,Slot1<R,C>,Slot4<C,P1,P2,P3,P4>);
    Func func=&chain4_<R,C,P1,P2,P3,P4>;
    return bind(slot(func),setter,getter);
  }


template <class R,class C,class P1,class P2,class P3,class P4,class P5> 
R chain5_(P1 p1,P2 p2,P3 p3,P4 p4,P5 p5,Slot1<R,C> setter,Slot5<C,P1,P2,P3,P4,P5> getter)
  { return setter(getter(p1,p2,p3,p4,p5)); } 
template <class R,class C,class P1,class P2,class P3,class P4,class P5> 
Slot5<R,P1,P2,P3,P4,P5> chain(const Slot1<R,C>& setter,const Slot5<C,P1,P2,P3,P4,P5>& getter) 
  { 
    typedef R (*Func)(P1,P2,P3,P4,P5,Slot1<R,C>,Slot5<C,P1,P2,P3,P4,P5>);
    Func func=&chain5_<R,C,P1,P2,P3,P4,P5>;
    return bind(slot(func),setter,getter);
  }



#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // SIGC_CHAIN_H
