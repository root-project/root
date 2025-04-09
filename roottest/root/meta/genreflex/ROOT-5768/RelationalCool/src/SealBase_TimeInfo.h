#ifndef SEAL_BASE_TIME_INFO_H
#define SEAL_BASE_TIME_INFO_H 1

#include "CoolKernel/types.h"
#include "CoolKernel/VersionInfo.h"

#ifdef COOL_ENABLE_TIMING_REPORT

namespace seal 
{
  //<<<<<< PUBLIC DEFINES                                                 >>>>>>
  //<<<<<< PUBLIC CONSTANTS                                               >>>>>>
  //<<<<<< PUBLIC TYPES                                                   >>>>>>
  //<<<<<< PUBLIC VARIABLES                                               >>>>>>
  //<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
  //<<<<<< CLASS DECLARATIONS                                             >>>>>>

  /** Utilities for monotonically growing high-resolution timers.
   *
   *  This class provides access to, among other things, virtual and
   *  real nanosecond-resolution timing info.  The implementation does
   *  its best to use the cheapest, most trustworthy time calculation
   *  method: system-provided high-resolution clocks or reading CPU
   *  cycle counters directly.  If those are not available, falls back
   *  to whatever is likely to produce the best data on the system,
   *  usually a system call that promises best resolution.
   *
   *  On systems that do provide accurate monotonic process-specific or
   *  system-wide high-resolution clocks (e.g. POSIX CLOCK_MONOTONIC and
   *  CLOCK_PROCESS_CPUTIME_ID clocks), they are used in preference to CPU
   *  cycle counters.  If no such clock is available, the monotonicity
   *  cannot always be guaranteed:
   *   - On a SMP system the process may hop from one processor to
   *     another, reading cycle counters on different CPUs.
   *   - Most multi-processor systems allow CPUs to be taken offline and
   *     put back online at any time.  The cycle counters may be reset
   *     or slowed down while the processor is offline.
   *   - Advanced power saving features can slow down CPU clock rates or
   *     put processes or the whole system to sleep or suspend mode.  In
   *     such a case the returned cycle counts may still be accurate but
   *     cannot be converted to nanoseconds meaningfully.  At any rate
   *     it is usually impossible to find out when this has happened.
   *     In fact the clock speed reported by the system may not even be
   *     right if something (e.g. the user) has put the system in a
   *     power-conserving mode that has slowed down the CPU clock rate
   *     -- the CPU clock rate may be wrong for the whole duration of
   *     the program.
   *
   *  In most of these cases it is anybody's guess what the timers read
   *  after such an event.  Most likely the readings are not linear.
   *
   *  The clock ticks are represented as a 64-bit signed integral type
   *  (see #NanoTicks).  Nanosecond times are represented as a double
   *  (see #NanoSecs).  This accomodates some 290 years worth of cycle
   *  counter ticks on a 1GHz CPU.  Cycle counters are usually zeroed on
   *  boot, so this should be plenty enough for another few years to
   *  come.  Please note however that not all systems provide cycle
   *  counters with this many significant bits.
   */
  class TimeInfo
  {
  public:
    // FIXME: feature bits for...?
    //  - whether real cycles are exact or derived
    //  - whether real nsecs are exact or derived
    //  - whether real nsecs were derived from cycles and mhz
    //  - whether real cycles were derived from nsecs and mhz
    //  - (the above four for virtual)
    //  - whether virtual nsecs were derived from process times
    //  - real/virtual resolution
    //  - sleep resolution

    /// #feature() bit indicating that #mhz() is the exact value
    /// provided by the system (for the cycle counts vs. nsecs).  If
    /// not set, the speed was estimated with a calibration loop.
    static const int FEATURE_EXACT_MHZ  = 1;

    /// #feature() bit indicating that #time() may not be
    /// process-specific but can have system-wide source.
    static const int FEATURE_TIME_EPOCH  = 2;

    /// #feature() bit indicating that #realCycles() and
    /// #realNsecs() can have have system-wide source.
    static const int FEATURE_REAL_COUNT_EPOCH = 4;

    /// #feature() bit indicating that #processUserTime(),
    /// #processSystemTime() and #processCpuTime() are
    /// meaningful.
    static const int FEATURE_PROCESS_TIMES = 16;

    /// Type for nanosecond times.
    typedef double NanoSecs;

    /// Type for cpu cycle counters.
    typedef cool::Int64 NanoTicks;

    static void  init (void);

    static double mhz (void);
    static double ghz (void);
    static unsigned features ();

    // FIXME: Wall clock/real time support?  This really is #Time.
    // Would be neat however if we can find out accurate process
    // start-up time.  Do we need more than just processTimes()?
    //
    // POSIX systems with clock_gettime() may provide CLOCK_REALTIME
    // (= wall), CLOCK_MONOTONIC (= real), CLOCK_PROCESS_CPUTIME_ID (=
    // virtual) and CLOCK_THREAD_CPUTIME_ID (= virtual thread-specific)
    // -- check.

    // FIXME: Provide estimate of clock read overhead?

    // high-res monotonic process time consumption
    static void  processTimes (NanoSecs &user, NanoSecs &system,
                               NanoSecs &real);
    static NanoSecs processUserTime (void);
    static NanoSecs processSystemTime (void);
    static NanoSecs processCpuTime (void);
    static NanoSecs processIdleTime (void);
    static NanoSecs processRealTime (void);

    // high-res system timer; not (necessarily) anchored to process
    // time but guaranteed to be monotonic; usually measures system
    // time since boot or something like that.
    static NanoSecs time (void);
  private:
    static bool s_initialised;
    static unsigned s_features;
    static double s_ghz;
    static double s_hiResFactor;
    static NanoSecs s_clockBase;
  };

  //<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>

}

#endif

#endif
