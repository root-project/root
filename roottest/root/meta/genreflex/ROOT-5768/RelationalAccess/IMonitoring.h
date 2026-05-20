// -*- C++ -*-
// $Id: IMonitoring.h,v 1.7 2006-03-31 13:32:16 rado Exp $
#ifndef RELATIONALACCESS_IMONITORING_H
#define RELATIONALACCESS_IMONITORING_H

#include <iosfwd>

namespace coral
{
  namespace monitor
  {
    enum Type
      {
        Info    = 0x00000001,
        Time    = (Info<<2),
        Warning = (Info<<3),
        Error   = (Info<<4),
        Config  = (Info<<5)
      };

    enum Level
      {
        Off     = 0,
        Minimal = Error,
        Default = Info | Error,
        Debug   = Info | Warning | Error   | Config,
        Trace   = Info | Time    | Warning | Error | Config
      };
  }

  /**
   * Class IMonitoring
   *
   * User-level Interface for controlling the client-side monitoring for the current session.
   */
  class IMonitoring
  {
  public:
    /**
     * Starts the client-side monitoring for the current session.
     * Throws a MonitoringServiceNotFoundException if there is no monitoring service available.
     */
    virtual void start( monitor::Level level = monitor::Default ) = 0;

    /**
     * Stops the client side monitoring.
     * Throws a monitoring exception if something went wrong.
     */
    virtual void stop() = 0;

    /**
     * Reports whatever has been gather by the monitoring service to an std::ostream.
     * Throws a MonitoringServiceNotFoundException if there is no monitoring service available.
     */
    virtual void reportToOutputStream( std::ostream& os ) const = 0;

    /**
     * Triggers the reporting of the underlying monitoring service.
     * Throws a MonitoringServiceNotFoundException if there is no monitoring service available.
     */
    virtual void report() const = 0;

    /**
     * Returns the status of the monitoring.
     */
    virtual bool isActive() const = 0;

  protected:
    /// Protected empty destructor
    virtual ~IMonitoring() {}
  };

}

#endif
