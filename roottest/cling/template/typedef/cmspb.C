/*
bash$ rootcint -f dict.cc -DGOOD cmspb.C Linkdef.h 
bash$ rootcint -f dict.cc cmspb.C Linkdef.h 
Internal error: G__mark_linked_tagnum() Illegal tagnum -1
Segmentation fault (core dumped)
*/

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


class SimTrack;
class SimEvent {

public:
  typedef Container<SimTrack>  track_container;
  typedef refc_ptr<track_container>  track_containerRef;
};

#pragma link C++ class SimEvent;



