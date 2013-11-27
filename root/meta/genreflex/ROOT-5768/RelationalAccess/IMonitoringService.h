// $Id: IMonitoringService.h,v 1.9 2008-04-04 09:21:07 rado Exp $
#ifndef RELATIONALACCESS_IMONITORINGSERVICE_H
#define RELATIONALACCESS_IMONITORINGSERVICE_H

#include "IMonitoring.h"
#include <string>

namespace coral
{

  class IMonitoringReporter;

  namespace monitor
  {

    enum Source
      {
        Application = 0x00000010,
        Session     = (Application<<2),
        Transaction = (Application<<3),
        Statement   = (Application<<4)
      };

    /**
     * Class IMonitoringService
     *
     * Developer-level interface for a monitoring service.
     * If any of the calls fails a MonitoringException is thrown.
     */

    class IMonitoringService
    {

    public:

      /**
       * Sets the level
       * @param contextKey The session ID for which to make the report
       * @param level      The monitoring level ( Default, Debug, Trace )
       */
      virtual void setLevel( const std::string& contextKey, coral::monitor::Level level ) = 0;

      /**
       * Return current monitoring level
       */
      virtual coral::monitor::Level level( const std::string& contextKey ) const = 0;

      /**
       * Return monitoring activity status
       */
      virtual bool active( const std::string& contextKey ) const = 0;

      /**
       * Enable monitoring for the given session
       */
      virtual void enable( const std::string& contextKey ) = 0;

      /**
       * Disable monitoring for the given session
       */
      virtual void disable( const std::string& contextKey ) = 0;

      /**
       * Records an event without a payload ( time event for example )
       */
      virtual void record( const std::string& contextKey,
                           Source source,
                           coral::monitor::Type type,
                           const std::string& description ) = 0;

      /**
       * Records an event with a payload
       */
      virtual void record( const std::string& contextKey,
                           Source source,
                           coral::monitor::Type type,
                           const std::string& description,
                           int data ) = 0;

      /**
       * Records an event with a payload
       */
      virtual void record( const std::string& contextKey,
                           Source source,
                           coral::monitor::Type type,
                           const std::string& description,
                           long long data ) = 0;

      /**
       * Records an event with a payload
       */
      virtual void record( const std::string& contextKey,
                           Source source,
                           coral::monitor::Type type,
                           const std::string& description,
                           double data ) = 0;

      /**
       * Records an event with a payload
       */
      virtual void record( const std::string& contextKey,
                           Source source,
                           coral::monitor::Type type,
                           const std::string& description,
                           const std::string& data ) = 0;

      /**
       * Return the cont reference to a reporting component which allows to
       * query & report the collected monitoring data
       */
      virtual const coral::IMonitoringReporter& reporter() const = 0;

      /**
       * Reports the events to the default reporter
       * @param contextKey The session ID for which to make the report
       */
      //virtual void report( const std::string& contextKey ) const = 0;

      /**
       * Reports the events to the specified output stream
       * @param contextKey The session ID for which to make the report
       */
      //virtual void reportToOutputStream( const std::string& contextKey, std::ostream& os ) const = 0;

    protected:

      /// Protected empty destructor
      virtual ~IMonitoringService() {}

    };

  }

}
#endif
