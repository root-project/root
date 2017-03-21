#ifndef ROO_TIMER
#define ROO_TIMER

#include <chrono>
#include <ctime>
#include "TSystem.h"
#include "TObject.h"
#include "Rtypes.h"

class RooTimer: public TObject {
public:
  RooTimer(); // default constructor necessary for ROOT classes to compile (https://root.cern.ch/root/Using.html)
  ~RooTimer(); // default destructor necessary to avoid "undefined reference to `vtable for RooTimer'"
  virtual void stop() = 0;
  double timing_s();
  void set_timing_s(double timing_s);

private:
  double _timing_s;

  ClassDef(RooTimer,1)
};

class RooInstantTimer: public RooTimer {
public:
  RooInstantTimer();
//  ~RooInstantTimer();
  virtual void stop();

  /* FIXME FIXME FIXME
   * DEZE VIRTUAL HIERBOVEN (EN HIERONDER) WEGHALEN?
   */

private:
  std::chrono::time_point<std::chrono::system_clock> _timing_begin, _timing_end;

  ClassDef(RooInstantTimer,1)
};


//class RooInstantCPUTimer: public RooTimer {
//public:
//  RooInstantCPUTimer();
//  virtual void stop();
//
//private:
//  struct timespec _timing_begin, _timing_end;
//
//  ClassDef(RooInstantCPUTimer,1)
//};

#endif
