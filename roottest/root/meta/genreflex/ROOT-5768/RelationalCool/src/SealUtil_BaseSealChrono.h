#ifndef SEALUTIL_BASESEALCHRONO_H
#define SEALUTIL_BASESEALCHRONO_H 1

#include <string>
#include <vector>
#include "CoolKernel/VersionInfo.h"

#ifdef COOL_ENABLE_TIMING_REPORT

namespace seal
{

  // Project   : LCG
  // Package   : SealUtil
  // Author    : Lorenzo.MONETA@cern.ch
  // Created by: moneta  at Tue Sep  2 16:03:40 2003
  
  /**
   * Basic class for Seal Chronos used by timers and TimingReport
   */
  class BaseSealChrono
  {

  public:

    virtual ~BaseSealChrono() {}


    virtual void start() = 0;

    virtual void stop() = 0;

    virtual std::vector<std::string> names() const = 0;

    virtual std::vector<double> values() const = 0;

    virtual unsigned int nTypes() const = 0;


    enum UnitType { NANOSECONDS = 0, SECONDS = 1, CLOCKTICKS = 2 };
    //values is nanoseconds

    virtual unsigned int timeUnit() { return BaseSealChrono::NANOSECONDS; }

  };

}

#endif

#endif
