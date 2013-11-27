#ifndef RELATIONALCOOL_COOLCHRONO_H
#define RELATIONALCOOL_COOLCHRONO_H 1

// Include files
#include <string>
#include <vector>
#include "CoolKernel/VersionInfo.h"
#include "SealUtil_BaseSealChrono.h"

#ifdef COOL_ENABLE_TIMING_REPORT

namespace cool
{

  /**
   * Time measurement performed using TimeInfo:
   *   measured Real, CPU (separated by User and System) and idle time
   */
  class CoolChrono : public seal::BaseSealChrono
  {

  public:

    CoolChrono();
    virtual ~CoolChrono() { /* no op */ }

    typedef double TimeUnit;

    enum  { realTime, userTime, systemTime, cpuTime, idleTime,
            vmSize, vmRss, n };

    std::vector<std::string> names() const { return m_names; }

    // return measured value
    std::vector<double> values() const { return m_values; }

    unsigned int nTypes() const { return n; }

    void start();
    void stop();

  private:

    std::vector<std::string> m_names;
    std::vector<double> m_values;
    bool m_started;

    // initial values
    TimeUnit m_realTime;
    TimeUnit m_userTime;
    TimeUnit m_sysTime;
    TimeUnit m_cpuTime;
    TimeUnit m_idleTime;
    long m_vmSize;
    long m_vmRss;

  };

}

#endif

#endif
