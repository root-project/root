#ifndef RELATIONALACCESS_SESSION_EXCEPTION_H
#define RELATIONALACCESS_SESSION_EXCEPTION_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

#include "CoralBase/Exception.h"

namespace coral
{

  /// Base exception class for errors related to a session.
  class SessionException : public Exception
  {
  public:
    /// Constructor
    SessionException( const std::string& message,
                      const std::string& methodName,
                      const std::string& moduleName ) :
      Exception( message, methodName, moduleName ) {}
    /// Default constructor
    SessionException() {}
    /// Destructor
    ~SessionException() throw() override {}
  };


  /// Exception thrown when the maximum number of sessions
  /// is exceeded per connection
  class MaximumNumberOfSessionsException : public SessionException
  {
  public:
    /// Constructor
    MaximumNumberOfSessionsException( const std::string& moduleName ) :
      SessionException( "Cannot create another session in the same connection",
                        "IConnection::newSession",
                        moduleName ) {}
    /// Default constructor
    MaximumNumberOfSessionsException() {}
    /// Destructor
    ~MaximumNumberOfSessionsException() throw() override {}
  };


  /// Exception thrown when a new session is refused by the server.
  /// It describes a possibly temporarely condition.
  class StartSessionException : public SessionException
  {
  public:
    /// Constructor
    StartSessionException( const std::string& moduleName,
                           const std::string& methodName ) :
      SessionException( "Cannot create a new session.",
                        methodName, moduleName ) {}
    /// Default constructor
    StartSessionException() {}
    /// Destructor
    ~StartSessionException() throw() override {}
  };


  class InvalidOperationInReadOnlyModeException : public SessionException
  {
  public:
#ifdef CORAL240EX
    /// Constructor
    InvalidOperationInReadOnlyModeException( const std::string& moduleName,
                                             const std::string& methodName ) :
      SessionException( "An update operation during a read-only session has been requested",
                        methodName, moduleName ) {}
#else
    /// Constructor
    InvalidOperationInReadOnlyModeException( const std::string& moduleName,
                                             const std::string& methodName ) :
      SessionException( "An unpdate operation during a read-only session has been requested",
                        methodName, moduleName ) {}
#endif
    /// Default constructor
    InvalidOperationInReadOnlyModeException() {}
    /// Destructor
    ~InvalidOperationInReadOnlyModeException() throw() override {}
  };


  /// Exception thrown when an invalid schema name exception is specified
  class InvalidSchemaNameException : public SessionException
  {
  public:
    /// Constructor
    InvalidSchemaNameException( const std::string& moduleName,
                                std::string methodName = "ISession::schema" ) :
      SessionException( "An invalid schema name has been specified",
                        methodName, moduleName ) {}
    /// Default constructor
    InvalidSchemaNameException() {}
    /// Destructor
    ~InvalidSchemaNameException() throw() override {}
  };


  /// Exception thrown when no authentication service has been loaded
  /// when a user session is started
  class NoAuthenticationServiceException : public SessionException
  {
  public:
    /// Constructor
    NoAuthenticationServiceException( const std::string& moduleName,
                                      std::string methodName = "ISession::startUserSession" ) :
      SessionException( "No Authentication Service has been loaded",
                        methodName, moduleName ) {}
    /// Default constructor
    NoAuthenticationServiceException() {}
    /// Destructor
    ~NoAuthenticationServiceException() throw() override {}
  };


  /// Exception thrown when authentication fails with a database server.
  class AuthenticationFailureException : public SessionException
  {
  public:
    /// Constructor
    AuthenticationFailureException( const std::string& moduleName,
                                    std::string methodName = "ISession::startUserSession" ) :
      SessionException( "Failed to authenticate with the server",
                        methodName, moduleName ) {}
    /// Default constructor
    AuthenticationFailureException() {}
    /// Destructor
    ~AuthenticationFailureException() throw() override {}
  };


  /// Exception thrown when a Monitoring Service has not been found
  /// in the context hierarchy
  class MonitoringServiceNotFoundException : public SessionException
  {
  public:
    /// Constructor
    MonitoringServiceNotFoundException( const std::string& moduleName,
                                        std::string methodName = "IMonitoring::start" ) :
      SessionException( "No Monitoring service has been found",
                        methodName, moduleName ) {}
    /// Default constructor
    MonitoringServiceNotFoundException() {}
    /// Destructor
    ~MonitoringServiceNotFoundException() throw() override {}
  };


#ifdef CORAL240EX
  /// Exception thrown when the user session is not active
  class SessionNotActiveException : public SessionException
  {
  public:
    /// Constructor
    SessionNotActiveException( const std::string& moduleName,
                               const std::string& methodName,
                               const std::string& message = "The user session is not active" ) :
      SessionException( moduleName, message, methodName ) {}
    /// Default constructor
    SessionNotActiveException() {}
    /// Destructor
    virtual ~SessionNotActiveException() throw() {}
  };
#endif


  /// Exception thrown when the database server cannot be reached.
  class ServerException : public SessionException
  {
  public:
    /// Constructor
    ServerException( const std::string& moduleName,
                     std::string message = "The database server could not be reached.",
                     std::string methodName = "ISession::connect" ) :
      SessionException( message, methodName, moduleName ) {}
    /// Default constructor
    ServerException() {}
    /// Destructor
    ~ServerException() throw() override {}
  };


  /// Exception thrown when the requested database server cannot be resolved
  /// or contacted.
  class DatabaseNotAccessibleException : public ServerException
  {
  public:
    /// Constructor
    DatabaseNotAccessibleException( const std::string& moduleName,
                                    std::string methodName,
                                    std::string message = "The specified service could not be reached."  ) :
      ServerException( moduleName, message, methodName ) {}
    /// Default constructor
    DatabaseNotAccessibleException() {}
    /// Destructor
    ~DatabaseNotAccessibleException() throw() override {}
  };


  /// Exception thrown when the database server cannot be reached.
  class ConnectionException : public ServerException
  {
  public:
    /// Constructor
    ConnectionException( const std::string& moduleName,
                         std::string methodName,
                         std::string message = "Cannot connect to the server."  ) :
      ServerException( moduleName, message, methodName ) {}
    /// Default constructor
    ConnectionException() {}
    /// Destructor
    ~ConnectionException() throw() override {}
  };


  /// Exception thrown when no connection to the database is active
  class ConnectionNotActiveException : public ConnectionException
  {
  public:
    /// Constructor
    ConnectionNotActiveException( const std::string& moduleName,
                                  std::string methodName = "ISession::userSchema",
                                  std::string message = "No connection to the server is active") :
      ConnectionException( moduleName, message, methodName ) {}
    /// Default constructor
    ConnectionNotActiveException() {}
    /// Destructor
    ~ConnectionNotActiveException() throw() override {}
  };


  /// Exception thrown when the connection was lost (network glitch)
  /// and reconnecting failed or was not possible.
  class ConnectionLostException : public ConnectionNotActiveException
  {
  public:
    /// Constructor
    ConnectionLostException( const std::string& moduleName,
                             std::string methodName,
                             std::string message = "" )
      : ConnectionNotActiveException( moduleName, "The connection was lost! " + message, methodName ) {}
    /// Default constructor
    ConnectionLostException() {}
    /// Destructor
    ~ConnectionLostException() throw() override {}
  };


  /// Base class for transaction-related exceptions
  class TransactionException : public SessionException
  {
  public:
    /// Constructor
    TransactionException( const std::string& moduleName,
                          const std::string& message,
                          const std::string& methodName ) :
      SessionException( message, methodName, moduleName ) {}
    /// Default constructor
    TransactionException() {}
    /// Destructor
    ~TransactionException() throw() override {}
  };


  /// Exception thrown when a transaction could not be started
  class TransactionNotStartedException : public TransactionException
  {
  public:
    /// Constructor
    TransactionNotStartedException( const std::string& moduleName,
                                    std::string message = "A transaction could not be started" ) :
      TransactionException( moduleName, message, "ITransaction::start" ) {}
    /// Default constructor
    TransactionNotStartedException() {}
    /// Destructor
    ~TransactionNotStartedException() throw() override {}
  };


  /// Exception thrown when a transaction could not be commited
  class TransactionNotCommittedException : public TransactionException
  {
  public:
    /// Constructor
    TransactionNotCommittedException( const std::string& moduleName,
                                      std::string message = "A transaction could not be committed" ) :
      TransactionException( moduleName, message, "ITransaction::commit" ) {}
    /// Default constructor
    TransactionNotCommittedException() {}
    /// Destructor
    ~TransactionNotCommittedException() throw() override {}
  };


  /// Exception thrown when a transaction is not active
  class TransactionNotActiveException : public TransactionException
  {
  public:
    /// Constructor
    TransactionNotActiveException( const std::string& moduleName,
                                   const std::string& methodName,
                                   std::string message = "A transaction is not active" ) :
      TransactionException( moduleName, message, methodName ) {}
    /// Default constructor
    TransactionNotActiveException() {}
    /// Destructor
    ~TransactionNotActiveException() throw() override {}
  };


  /// Exception thrown when an update operation is attempted during
  /// a read-only transaction
  class InvalidOperationInReadOnlyTransactionException : public SessionException
  {
  public:
    /// Constructor
    InvalidOperationInReadOnlyTransactionException( const std::string& moduleName,
                                                    const std::string& methodName ) :
      SessionException( "An update operation during a read-only transaction has been requested",
                        methodName, moduleName ) {}
    /// Default constructor
    InvalidOperationInReadOnlyTransactionException() {}
    /// Destructor
    ~InvalidOperationInReadOnlyTransactionException() throw() override {}
  };


  /// Base class for type-converter-related exceptions
  class TypeConverterException : public SessionException
  {
  public:
    /// Constructor
    TypeConverterException( const std::string& moduleName,
                            const std::string& message,
                            const std::string& methodName ) :
      SessionException( message,
                        "ITypeConverter::" + methodName,
                        moduleName ) {}
    /// Default constructor
    TypeConverterException() {}
    /// Destructor
    ~TypeConverterException() throw() override {}
  };


  /// Exception thrown whenever an SQL type is not supported
  class UnSupportedSqlTypeException : public TypeConverterException
  {
  public:
    /// Constructor
    UnSupportedSqlTypeException( const std::string& moduleName,
                                 const std::string& methodName,
                                 const std::string& sqlType ) :
      TypeConverterException( moduleName,
                              "SQL type \"" + sqlType + "\" is not supported",
                              methodName ) {}
    /// Default constructor
    UnSupportedSqlTypeException() {}
    /// Destructor
    ~UnSupportedSqlTypeException() throw() override {}
  };


  /// Exception thrown whenever a C++ type is not supported
  class UnSupportedCppTypeException : public TypeConverterException
  {
  public:
    /// Constructor
    UnSupportedCppTypeException( const std::string& moduleName,
                                 const std::string& methodName,
                                 const std::string& cppType ) :
      TypeConverterException( moduleName,
                              "C++ type \"" + cppType + "\" is not supported",
                              methodName ) {}
    /// Default constructor
    UnSupportedCppTypeException() {}
    /// Destructor
    ~UnSupportedCppTypeException() throw() override {}
  };


}

#endif
