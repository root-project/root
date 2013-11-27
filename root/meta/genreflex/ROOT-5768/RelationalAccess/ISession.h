#ifndef RELATIONALACCESS_ISESSION_H
#define RELATIONALACCESS_ISESSION_H

#include <string>

namespace coral {

  // forward declarations
  class IMonitoring;
  class ITransaction;
  class ISchema;

  /**
   * Class ISession
   * Interface class responsible for establishing the physical connection to a database,
   * authenticating and controlling the client-side monitoring.
   */
  class ISession {
  public:
    /// Destructor. It releases the reference to the parent Connection.
    virtual ~ISession() {}

    /**
     * Returns the reference to the underlying IMonitoring object.
     */
    virtual IMonitoring& monitoring() = 0;

    /**
     * Authenticates with the database server using a user/password pair.
     * If the authentication fails an AuthenticationFailureException is thrown.
     * If a new session cannot start but only temporarily, a StartSessionException is thrown,
     * else a SessionException is thrown.
     */
    virtual void startUserSession( const std::string& userName,
                                   const std::string& password ) = 0;

    /**
     * Terminates a user session without dropping the connection to the database server
     */
    virtual void endUserSession() = 0;

    /**
     * Returns the status of a user session.
     */
    virtual bool isUserSessionActive() const = 0;

    /**
     * Returns the corresponding ITransaction object.
     * If a connection is not yet established, a ConnectionNotActiveException is thrown.
     */
    virtual ITransaction& transaction() = 0;

    /**
     * Returns a reference to the working ISchema object.
     * If a connection is not yet established, a ConnectionNotActiveException is thrown.
     */
    virtual ISchema& nominalSchema() = 0;

    /**
     * Returns a reference to the ISchema object corresponding to the specified name.
     * If a connection is not yet established, a ConnectionNotActiveException is thrown.
     * If no schema exists corresponding to the specified name an InvalidSchemaNameException is thrown.
     */
    virtual ISchema& schema( const std::string& schemaName ) = 0;

    /**
     * Returns the technology name for the remote session.
     * For plugins establishing a database connection through a middle-tier
     * (e.g. CoralAccess), this is discovered when establishing the remote session.
     */
    virtual std::string remoteTechnologyName() const = 0;

    /**
     * Returns the technology name for the remote session.
     * For plugins establishing a database connection through a middle-tier
     * (e.g. CoralAccess), this is discovered when establishing the remote session.
     */
    virtual std::string remoteServerVersion() const = 0;

  };

}

#endif
