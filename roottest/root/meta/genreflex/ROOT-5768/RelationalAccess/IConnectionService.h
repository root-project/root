#ifndef RELATIONALACCESS_ICONNECTIONSERVICE_H
#define RELATIONALACCESS_ICONNECTIONSERVICE_H

#include "AccessMode.h"

#include <string>

namespace coral {

  // forward declarations
  class ISessionProxy;
  class IConnectionServiceConfiguration;
  class IMonitoringReporter;
  class IWebCacheControl;

  /**
   * Class IConnectionService
   *
   * Interface for a facade to the low level connection-related classes of CORAL,
   * which takes care of the proper application-level database connection management.
   * A user retrieves a fully operational connection to a database by simply specifying
   * the connection name (which can be either physical or logical) and the access mode.
   *
   * Any implementation of the interface should be a seal service, which should load
   * in its current context (if missing from its context hierarchy) the necessary
   * relational, monitoring, lookup and authentication services.
   */
  class IConnectionService
  {
  public:
    /**
     * Returns a session proxy object for the specified connection string
     * and access mode.
     */
    virtual ISessionProxy* connect( const std::string& connectionName,
                                    AccessMode accessMode = Update ) = 0;

    /**
     * Returns a session proxy object for the specified connection string, role
     * and access mode.
     */
    virtual ISessionProxy* connect( const std::string& connectionName,
                                    const std::string& asRole,
                                    AccessMode accessMode = Update ) = 0;
    /**
     * Returns the configuration object for the service.
     */
    virtual IConnectionServiceConfiguration& configuration() = 0;

    /**
     * Cleans up the connection pool from the unused connection, according to
     * the policy defined in the configuration.
     */
    virtual void purgeConnectionPool() = 0;

    /**
     * Returns the monitoring reporter
     */
    virtual const IMonitoringReporter& monitoringReporter() const = 0;

    /**
     * Returns the object which controls the web cache
     */
    virtual IWebCacheControl& webCacheControl() = 0;

  protected:
    /// Protected empty destructor
    virtual ~IConnectionService() {}
  };

}

#endif
