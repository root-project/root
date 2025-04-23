#ifndef SEALUTIL_SEALTIMER_H
#define SEALUTIL_SEALTIMER_H 1

#include <iostream>
#include <string>
#include "CoolKernel/VersionInfo.h"

#ifdef COOL_ENABLE_TIMING_REPORT

namespace seal
{

  class TimingItem;
  class BaseSealChrono;

  /**
   * Seal timer class measuring real time, CPU time (separated as user and system)
   * and idle time
   * Timer starts when constructed and finish at destructions
   *
   * NOTE: This class does not support copying.
   */

  // Project   : LCG
  // Package   : SealUtil
  // Author    : Lorenzo.MONETA@cern.ch
  // Created by: moneta  at Fri Aug 29 16:05:03 2003
  
  class SealTimer
  {

  public:
    // construct using default chrono (SealBaseChrono)
    //SealTimer(const std::string & s = "", bool printResult = true, std::ostream & out = std::cout);
    // constructors passing a chrono and a string - no connection to report
    SealTimer(BaseSealChrono & c, const std::string & s = "", bool printResult = true, std::ostream & out = std::cout);
    // construct from item
    SealTimer(TimingItem & item, bool printResult = false, std::ostream & out = std::cout);
    virtual ~SealTimer();

  private:
    /// copying unimplemented in this class.
    SealTimer(const SealTimer&) : m_out (std::cout) { }
    /// copying unimplemented in this class.
    SealTimer & operator = (const SealTimer & rhs) {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
    }

  public:

    /*
   * start measuring time. Normally not need to call, it is called automatically
   *  in constructors
   */

    void start();

    /*
   * stop timer and calculate difference from start.
   * called automatically in destructor
   */

    void stop();

    /*
   *  print timing result. Called also automatically from destructors
   */

    void print();

    /*
   * return elapsed time since start.
   * Index i correspond to type of time identified
   *  according to the chrono used
   */

    double elapsed(unsigned int i = 0);

  protected:


  private:

    // initial times in nano-seconds


    bool m_printResult;
    std::ostream & m_out;

    TimingItem * m_item;
    bool m_ownItem;
    BaseSealChrono * m_default_chrono;

  };

}

#endif

#endif
