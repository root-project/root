#ifndef RELATIONALACCESS_IAUTHENTICATIONSERVICE_H
#define RELATIONALACCESS_IAUTHENTICATIONSERVICE_H

#include <string>

namespace coral {

  // forward declarations
  class IAuthenticationCredentials;

  /**
   * Class IAuthenticationService
   * Interface for an authentication service. Implementations should be seal services.
   * Its role is to provide the necessary authentication credentials given a connection string.
   */
  class IAuthenticationService {
  public:
    /**
     * Returns a reference to the credentials object for a given connection string.
     * If the connection string is not known to the service an UnknownConnectionException is thrown.
     */
    virtual const IAuthenticationCredentials& credentials( const std::string& connectionString ) const = 0;

    /**
     * Returns a reference to the credentials object for a given connection string and a "role".
     * If the connection string is not known to the service an UnknownConnectionException is thrown.
     * If the role specified does not exist an UnknownRoleException is thrown.
     */
    virtual const IAuthenticationCredentials& credentials( const std::string& connectionString,
                                                           const std::string& role ) const = 0;
  protected:
    /// Protected empty destructor
    virtual ~IAuthenticationService() {}
  };

}

#endif
