#ifndef RELATIONALACCESS_AUTHENTICATIONSERVICE_EXCEPTION_H
#define RELATIONALACCESS_AUTHENTICATIONSERVICE_EXCEPTION_H

#include "CoralBase/Exception.h"

namespace coral {

  /**
   * Class AuthenticationServiceException
   *
   * Base exception class for the errors related to / produced by an
   * IAuthenticationService implementation.
   */

  class AuthenticationServiceException : public Exception
  {
  public:
    /// Constructor
    AuthenticationServiceException( const std::string& message,
                                    const std::string& methodName,
                                    const std::string& moduleName ) :
      Exception( message, methodName, moduleName )
    {}

    /// Destructor
    ~AuthenticationServiceException() throw() override {}
  };



  /// Exception class thrown when an unknown connection is specified to the authentication service.
  class UnknownConnectionException : public AuthenticationServiceException
  {
  public:
    /// Constructor
    UnknownConnectionException( const std::string& moduleName,
                                const std::string& connectionName,
                                std::string methodName = "IAuthenticationService::credentials" ) :
      AuthenticationServiceException( "Connection string \"" + connectionName + "\" is not known to the service",
                                      methodName, moduleName )
    {}

    /// Destructor
    ~UnknownConnectionException() throw() override {}
  };



  /// Exception class thrown when an unknown role is specified to the authentication service.
  class UnknownRoleException : public AuthenticationServiceException
  {
  public:
    /// Constructor
    UnknownRoleException( const std::string& moduleName,
                          const std::string& connectionName,
                          const std::string& role,
                          std::string methodName = "IAuthenticationService::credentials" ) :
      AuthenticationServiceException( "Role \"" + role + "\" is not known for connection string \"" + connectionName + "\"",
                                      methodName, moduleName )
    {}

    /// Destructor
    ~UnknownRoleException() throw() override {}
  };



  /// Exception class thrown when an invalid name is specified when asking to retrieve a credential item
  class InvalidCredentialItemException : public AuthenticationServiceException
  {
  public:
    /// Constructor
    InvalidCredentialItemException( const std::string& moduleName,
                                    const std::string& itemName,
                                    std::string methodName = "IAuthenticationCredentials::valueForItem" ) :
      AuthenticationServiceException( "Authentication credential item \"" + itemName + "\" is not known",
                                      methodName, moduleName )
    {}

    /// Destructor
    ~InvalidCredentialItemException() throw() override {}
  };



  /// Exception class thrown when an invalid index is specified when asking to retrieve a credential item
  class CredentialItemIndexOutOfRangeException : public AuthenticationServiceException
  {
  public:
    /// Constructor
    CredentialItemIndexOutOfRangeException( const std::string& moduleName,
                                            std::string methodName = "IAuthenticationCredentials::itemName" ) :
      AuthenticationServiceException( "Invalid credential item index has been specified",
                                      methodName, moduleName )
    {}

    /// Destructor
    ~CredentialItemIndexOutOfRangeException() throw() override {}
  };

}

#endif
