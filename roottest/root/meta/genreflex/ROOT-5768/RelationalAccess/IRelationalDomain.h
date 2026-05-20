#ifndef RELATIONALACCESS_IRELATIONALDOMAIN_H
#define RELATIONALACCESS_IRELATIONALDOMAIN_H

#include <string>
#include <utility>

namespace coral {

  // forward declarations
  class IConnection;

  /**
   * Class IRelationalDomain
   * Interface for the entry point of an RDBMS-specific implementation.
   * Implementations of this interface should be seal services.
   */
  class IRelationalDomain {
  public:
    /**
     * Returns the name of the RDBMS flavour
     */
    virtual std::string flavorName() const = 0;

    /**
     * Returns the name of the implementation for the current module
     */
    virtual std::string implementationName() const = 0;

    /**
     * Returns the version name of the underlying implementation
     */
    virtual std::string implementationVersion() const = 0;

    /**
     * Returns a new connection object
     */
    virtual IConnection* newConnection( const std::string& connectionString ) const = 0;

    /**
     * Decodes the user connection string into the real connection string that should be passed to the Connection class
     * and the schema name that should be passed to the Session class
     */
    virtual std::pair<std::string, std::string > decodeUserConnectionString( const std::string& userConnectionString ) const = 0;

    /**
     * Returns true if credentials have to be provided to start a session on the specified connection
     */
    virtual bool isAuthenticationRequired( const std::string& connectionString ) const = 0;

  protected:
    /// Protected empty destructor
    virtual ~IRelationalDomain() {}
  };

}

#endif
