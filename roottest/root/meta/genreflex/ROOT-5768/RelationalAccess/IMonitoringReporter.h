// -*- C++ -*-
// $Id: IMonitoringReporter.h,v 1.1 2006-03-31 13:34:54 rado Exp $
#ifndef RELATIONALACCESS_IMONITORINGREPORTER_H
#define RELATIONALACCESS_IMONITORINGREPORTER_H 1

#include "IMonitoring.h"

#include <set>
#include <string>

namespace coral
{
  /**
   * Class IMonitoringReporter
   *
   * User-level interface for the client side monitoring system.
   * If any of the calls fails a MonitoringException is thrown.
   */

  class IMonitoringReporter
  {
  public:
    /**
     * Return the set of currently monitored data sources
     */
    virtual std::set< std::string > monitoredDataSources() const = 0;

    /**
     * Reports the events for all data sources being monitored
     * @param level      The OR-ed selection of even types to be reported
     */
    virtual void report( unsigned int level=coral::monitor::Default  ) const = 0;

    /**
     * Reports the events for a given data source name of a given monitoring level
     * @param contextKey The session ID for which to make the report
     * @param level      The OR-ed selection of even types to be reported
     */
    virtual void report( const std::string& contextKey,
                         unsigned int level=coral::monitor::Default  ) const = 0;

    /**
     * Reports the events for a given data source name to the specified output stream
     * @param contextKey The session ID for which to make the report
     * @param os         The output stream
     * @param level      The OR-ed selection of even types to be reported
     */
    virtual void reportToOutputStream( const std::string& contextKey, std::ostream& os,
                                       unsigned int level=coral::monitor::Default ) const = 0;

  protected:
    /// Protected empty destructor
    virtual ~IMonitoringReporter() {}
  };
}

#endif // RELATIONALACCESS_IMONITORINGREPORTER_H
