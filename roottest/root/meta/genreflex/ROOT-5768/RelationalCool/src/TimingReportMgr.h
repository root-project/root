// $Id: TimingReportMgr.h,v 1.5 2012-06-29 14:46:04 avalassi Exp $
#ifndef RELATIONALCOOL_TIMINGREPORTMGR_H
#define RELATIONALCOOL_TIMINGREPORTMGR_H 1

// Include files
#include <map>
#include "CoolKernel/VersionInfo.h"

// Local include files
#include "SealUtil_SealTimer.h"
#include "TimingReport.h"

namespace cool
{

  class TimingReportMgr 
  {

  public:

    static bool isActive();
    static void initialize();
    static void finalize();
    static void startTimer( const std::string& name );
    static void stopTimer( const std::string& name );

#ifdef COOL_ENABLE_TIMING_REPORT

  private:

    TimingReportMgr( );
    TimingReportMgr( const TimingReportMgr& rhs );
    TimingReportMgr& operator= ( const TimingReportMgr& rhs );
    virtual ~TimingReportMgr( );

  private:

    static TimingReport*& pTimingReport();
    static TimingReport& timingReport();
    typedef std::map< std::string, seal::SealTimer* > TimerMap;
    static TimerMap& timerMap();

#endif

  };

}
#endif // RELATIONALCOOL_TIMINGREPORTMGR_H
