#ifndef RELATIONALACCESS_CONNECTIONSERVICE_EXCEPTION_H
#define RELATIONALACCESS_CONNECTIONSERVICE_EXCEPTION_H

#include "CoralBase/Exception.h"
#include "RelationalAccess/AccessMode.h"

namespace coral {

  /**
   * Class ConnectionServiceException
   *
   * Base exception class for the errors related to / produced by an
   * IConnectionService implementation.
   */

  class ConnectionServiceException : public Exception
  {
  public:
    /// Constructor
    ConnectionServiceException( const std::string& message,
                                const std::string& methodName,
                                std::string moduleName = "CORAL/Services/ConnectionService" ) :
      Exception( message, methodName, moduleName )
    {}

    /// Destructor
    ~ConnectionServiceException() throw() override {}
  };



  /// Exception thrown when no connection trial was successfull.
  class ConnectionNotAvailableException : public ConnectionServiceException
  {
  public:
    /// Constructor
    ConnectionNotAvailableException( const std::string& connectionName,
                                     const std::string& methodName,
                                     std::string moduleName = "CORAL/Services/ConnectionService" ) :
      ConnectionServiceException( "Connection on \"" + connectionName + "\" cannot be established",
                                  methodName, moduleName ){}

    /// Destructor
    ~ConnectionNotAvailableException() throw() override {}
  };



  /// Exception thrown when no replica is available for a logical connection string
  class ReplicaNotAvailableException : public ConnectionServiceException
  {
  public:
    /// Constructor
    ReplicaNotAvailableException( const std::string& connectionName,
                                  coral::AccessMode accessMode,
                                  const std::string& methodName,
                                  std::string moduleName = "CORAL/Services/ConnectionService") :
      ConnectionServiceException( "No physical "+((accessMode==coral::ReadOnly) ? std::string("Read-Only") : std::string("Update"))+" connection for \"" + connectionName + "\" is available.",
                                  methodName, moduleName ){}

    /// Destructor
    ~ReplicaNotAvailableException() throw() override {}
  };

}

#endif
