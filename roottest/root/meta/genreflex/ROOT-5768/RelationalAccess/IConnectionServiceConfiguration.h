#ifndef RELATIONALACCESS_ICONNECTIONSERVICECONFIGURATION_H
#define RELATIONALACCESS_ICONNECTIONSERVICECONFIGURATION_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

#include <string>
#include "IMonitoring.h"

namespace coral
{

  class IAuthenticationService;
  class ILookupService;
  class IRelationalService;
  class IReplicaSortingAlgorithm;

  namespace monitor
  {
    class IMonitoringService;
  }


  /**
   * Class IConnectionServiceConfiguration
   *
   * Interface for configuring the connection service
   */
  class IConnectionServiceConfiguration
  {
  public:

    /**
     * Enables the failing over to the next available
     * replica in case the current one is not available.
     * Otherwise the ConnectionService gives up.
     */
    virtual void enableReplicaFailOver() = 0;

    /**
     * Disables the failing over to the next available
     * replica in case the current one is not available.
     */
    virtual void disableReplicaFailOver() = 0;

    /**
     * Returns the failover mode
     */
    virtual bool isReplicaFailoverEnabled() const = 0;

    /**
     * Enables the sharing of the same physical connection
     * among more clients.
     */
    virtual void enableConnectionSharing() = 0;

    /**
     * Disables the sharing of the same physical connection
     * among more clients.
     */
    virtual void disableConnectionSharing() = 0;

    /**
     * Returns true if the connction sharing is enabled
     */
    virtual bool isConnectionSharingEnabled() const = 0;

    /**
     * Enables the re-use of Update connections for Read-Only sessions
     */
    virtual void enableReadOnlySessionOnUpdateConnections() = 0;

    /**
     * Disables the re-use of Update connections for Read-Only sessions
     */
    virtual void disableReadOnlySessionOnUpdateConnections() = 0;

    /**
     * Returns true if the  re-use of Update connections for Read-Only sessions is enabled
     */
    virtual bool isReadOnlySessionOnUpdateConnectionsEnabled() const = 0;

    /**
     * Sets the period of connection retrials (time interval between two retrials).
     */
    virtual void setConnectionRetrialPeriod( int timeInSeconds ) = 0;

    /**
     * Returns the rate of connection retrials (time interval between two retrials).
     */
    virtual int connectionRetrialPeriod() const = 0;

    /**
     * Sets the time out for the connection retrials before the connection
     * service fails over to the next available replica or quits.
     */
    virtual void setConnectionRetrialTimeOut( int timeOutInSeconds ) = 0;

    /**
     * Returns the time out for the connection retrials before the connection
     * service fails over to the next available replica or quits.
     */
    virtual int connectionRetrialTimeOut() const = 0;

    /**
     * Sets the connection time out in seconds.
     */
    virtual void setConnectionTimeOut( int timeOutInSeconds ) = 0;

    /**
     * Retrieves the connection time out in seconds.
     */
    virtual int connectionTimeOut() const = 0;

    /**
     * Activate the parallel thread for idle pool cleaning up
     */
    virtual void enablePoolAutomaticCleanUp() = 0;

    /**
     * Disable the parallel thread for idle pool cleaning up
     */
    virtual void disablePoolAutomaticCleanUp() = 0;

    /**
     * Returns true if the parallel thread for idle pool cleaning up is enabled
     */
    virtual bool isPoolAutomaticCleanUpEnabled() const = 0;

    /**
     * Sets the time duration of exclusion from failover list for a
     * connection not available.
     */
    virtual void setMissingConnectionExclusionTime( int timeInSeconds ) = 0;

    /**
     * Retrieves the time duration of exclusion from failover list for a
     * connection not available.
     */
    virtual int missingConnectionExclusionTime() const = 0;

    /**
     * Sets the monitoring level for the new sessions.
     */
    virtual void setMonitoringLevel( monitor::Level level ) = 0;

    /**
     * Retrieves the current monitoring level.
     */
    virtual monitor::Level monitoringLevel() const = 0;

    /**
     * Loads and sets the authentication service to be used for the new sessions.
     */
    virtual void setAuthenticationService( const std::string serviceName ) = 0;

    /**
     * Loads and sets the default lookup service to be used for the new sessions.
     */
    virtual void setLookupService( const std::string& serviceName ) = 0;

    /**
     * Loads and sets the default relational service to be used for the new sessions.
     */
    virtual void setRelationalService( const std::string& serviceName ) = 0;

    /**
     * Loads and sets the default monitoring service to be used for the new sessions.
     */
    virtual void setMonitoringService( const std::string& serviceName ) = 0;

    /**
     * Loads and sets the authentication service to be used for the new sessions.
     */
    virtual void setAuthenticationService( IAuthenticationService& customAuthenticationService ) = 0;

    /**
     * Loads and sets the default lookup service to be used for the new sessions.
     */
    virtual void setLookupService( ILookupService& customLookupService ) = 0;

    /**
     * Loads and sets the default relational service to be used for the new sessions.
     */
    virtual void setRelationalService( IRelationalService& customRelationalService ) = 0;

    /**
     * Loads and sets the default monitoring service to be used for the new sessions.
     */
    virtual void setMonitoringService( monitor::IMonitoringService& customMonitoringService ) = 0;

    /**
     * Sets the algorithm to be used for the database replica ordering.
     */
    virtual void setReplicaSortingAlgorithm( IReplicaSortingAlgorithm& algorithm ) = 0;

#ifdef CORAL240CC
    /**
     * Get the authentication service (API extension bug #100862)
     */
    virtual IAuthenticationService& authenticationService() const = 0;

    /**
     * Get the lookup service (API extension bug #100862)
     */
    virtual ILookupService& lookupService() const = 0;

    /**
     * Get the relational service (API extension bug #100862)
     */
    virtual IRelationalService& relationalService() const = 0;

    /**
     * Get the monitoring service (API extension bug #100862)
     */
    virtual monitor::IMonitoringService& monitoringService() const = 0;

    /**
     * Get the authentication service (API extension bug #100862)
     */
    virtual IReplicaSortingAlgorithm* replicaSortingAlgorithm() const = 0;
#endif

  protected:
    /// Protected empty destructor
    virtual ~IConnectionServiceConfiguration() {}
  };

}

#endif
