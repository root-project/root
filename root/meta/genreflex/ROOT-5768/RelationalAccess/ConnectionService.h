#ifndef RELATIONALACCESS_CONNECTIONSERVICE_H
#define RELATIONALACCESS_CONNECTIONSERVICE_H

#include "IConnectionService.h"

#include "CoralBase/MessageStream.h"

namespace coral
{

  namespace Pimpl
  {
    class Holder;
  }
  /**
   * Class ConnectionService
   *
   * Proxy implementation for the IConnectionService interface, hidding all the complexities of the SEAL component model
   */
  class ConnectionService : virtual public IConnectionService {
  public:
    /// Constructor initialized with a context. If not specified a new context is created internally
    ConnectionService();

    /// Destructor
    ~ConnectionService() override;

    /**
     * Returns a session proxy object for the specified connection string
     * and access mode.
     */
    ISessionProxy* connect( const std::string& connectionName,
                            AccessMode accessMode = Update ) override;

    /**
     * Returns a session proxy object for the specified connection string, role
     * and access mode.
     */
    ISessionProxy* connect( const std::string& connectionName,
                            const std::string& asRole,
                            AccessMode accessMode = Update ) override;
    /**
     * Returns the configuration object for the service.
     */
    IConnectionServiceConfiguration& configuration() override;

    /**
     * Cleans up the connection pool from the unused connection, according to
     * the policy defined in the configuration.
     */
    void purgeConnectionPool() override;

    /**
     * Returns the monitoring reporter
     */
    const IMonitoringReporter& monitoringReporter() const override;

    /**
     * Returns the object which controls the web cache
     */
    IWebCacheControl& webCacheControl() override;

    // Set the user defined verbosity level
    void setMessageVerbosityLevel( coral::MsgLevel level );

  protected:
    // Trigger dynamic loading of the real ConnectionService when needed
    void loadConnectionService() const;

  private:
    /// No copy constructor or assignment operator
    ConnectionService( const ConnectionService& );
    ConnectionService& operator=( const ConnectionService& );

  private:
    mutable coral::Pimpl::Holder*  m_connectionService;
  };

}

#endif
