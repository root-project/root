#ifndef SEALUTIL_TIMINGITEM_H
#define SEALUTIL_TIMINGITEM_H 1

#include "CoolKernel/VersionInfo.h"
#include "SealUtil_BaseSealChrono.h"

#ifdef COOL_ENABLE_TIMING_REPORT

namespace seal
{

  /**
   * class used to collect the timing result and statistics for a specific item
   * Used by the TimingReport
   * This class does not support copying.
   */

  //template <class Chrono=SealChrono> class TimingItem;

  // Project   : LCG
  // Package   : SealUtil
  // Author    : Lorenzo.MONETA@cern.ch
  // Created by: moneta  at Tue Sep  2 14:06:56 2003

  class TimingItem
  {



    static const long nsec_per_sec = 1000000000; // nanosec in one second

    typedef BaseSealChrono Chrono;

  public:

    TimingItem(Chrono & c, const std::string & name);

    ~TimingItem() { /* no op */ }

  private:
    /// copying unimplemented in this class.
    TimingItem(const TimingItem &);
    /// copying unimplemented in this class.
    TimingItem & operator = (const TimingItem &);

  public:



    void accumulate();


    std::string name() { return m_name; }

    std::string unit() { return m_unit; }

    Chrono & chrono() { return m_chrono; }

    unsigned int numberOfMeasurements() const { return m_counter; }

    unsigned int numberOfTypes() const { return m_chrono.nTypes(); }

    std::string timeType(unsigned int i = 0) const;

    // unit here are doubles indicating the number of second

    double mean(unsigned int i = 0) const;

    double rms(unsigned int i = 0)  const;

    double lastValue(unsigned int i=0) const;


  private:

    std::string m_name;
    std::string m_unit;
    unsigned int m_counter;
    std::vector<double> m_sumV;
    std::vector<double> m_sumV2;
    // by default results are in seconds
    // conversion from nanosec to seconds
    double m_convScale;

    Chrono & m_chrono;

  };

}

#endif

#endif
