#ifndef RELATIONALACCESS_ICONNECTION_H
#define RELATIONALACCESS_ICONNECTION_H

#include <string>
#include "AccessMode.h"

namespace coral {

  // forward declarations
  class ISession;
  class ITypeConverter;

  /**
   * Class IConnection
   * Interface for a physical connection to a database.
   */
  class IConnection {
  public:
    /// Empty destructor
    virtual ~IConnection() {}

    /**
     * Connects to a database server without authenticating
     * If no connection to the server can be established a ServerException is thrown.
     * If no connection to the server can be established but only temporarily, a ConnectionException is thrown.
     * If a connection cannot be established for sure (eg. wrong connection parameters), a DatabaseNotAccessibleException is thrown.
     */
    virtual void connect() = 0;

    /**
     * Returns a new session object.
     * In case no more sessions can be established for the current physical connection,
     * a MaximumNumberOfSessionsException is thrown.
     */
    virtual ISession* newSession( const std::string& schemaName,
                                  coral::AccessMode mode = coral::Update ) const = 0;

    /**
     * Returns the connection status. By default this is a logical operation.
     * One should pass true as an argument to force the probing of the physical connection as well.
     */
    virtual bool isConnected( bool probePhysicalConnection = false ) = 0;

    /**
     * Drops the physical connection with the database.
     */
    virtual void disconnect() = 0;

    /**
     * Returns the version of the database server.
     * If a connection is not yet established, a ConnectionNotActiveException is thrown.
     */
    virtual std::string serverVersion() const = 0;

    /**
     * Returns the C++ <-> SQL type converter relevant to he current session.
     * If a connection is not yet established, a ConnectionNotActiveException is thrown.
     */
    virtual ITypeConverter& typeConverter() = 0;

  };

}

#endif
