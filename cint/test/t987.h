/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

// with cmspb.C
#ifndef GOOD
#include <vector>
#endif

template<class T>
class Container {
  //some container implementation
};

template <class X, bool Intr=false> class refc_ptr {
  //something totally generic
};

template <class X> class refc_ptr<X, false> {
  //something specific to the false case
};

template <class X> class refc_ptr<X,true> {
  //something specific to the true case
};


class SimTrack {};
class SimEvent {

public:
  typedef Container<SimTrack>  track_container;
  typedef refc_ptr<track_container>  track_containerRef;
};

#ifdef __MAKECINT__
#pragma link C++ class SimEvent;
#endif

