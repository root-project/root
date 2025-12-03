#ifndef RELATIONALACCESS_ISESSIONPROXY_H
#define RELATIONALACCESS_ISESSIONPROXY_H

#include <string>

namespace coral {

  // forward declarations
  class ISchema;
  class ITransaction;
  class ITypeConverter;
  class ISessionProperties;

  /**
   * Class ISessionProxy
   *
   * Interface for a proxy class to an ISession object.
   * Whenever the user retrieves the schema, transaction or
   * typeconverter objects, the connection is physically probed
   * and if necessary (and being outside of an update transaction)
   * a new connection is established.
   * In case the connection has been broken in the middle of an
   * update transaction, an exception is raised.
   */
  class ISessionProxy
  {
  public:
    /**
     * Virtual destructor
     */
    virtual ~ISessionProxy() {}

    /**
     * Returns the properties of the session opened in the local client plugin.
     * Since the underlying connection is probed before the object info construction, it always contains vaild data.
     */
    virtual ISessionProperties& properties() = 0;

    /**
     * Returns the properties of the remote session.
     * For plugins establishing a direct database connection (e.g. Oracle, MySQL, SQLite, Frontier),
     * these coincide with the 'local' session properties.
     * For plugins establishing a database connection through a middle-tier (e.g. CoralAccess),
     * these are different from the 'local' session properties.
     * In the latter case, an exception is thrown if no connection to the remote database is established yet.
     */
    virtual const ISessionProperties& remoteProperties() = 0;

    /**
     * Returns the working schema of the connection.
     */
    virtual ISchema& nominalSchema() = 0;

    /**
     * Returns a reference to the ISchema object corresponding to the specified name.
     */
    virtual ISchema& schema( const std::string& schemaName ) = 0;

    /**
     * Returns the transaction handle of the active session.
     */
    virtual ITransaction& transaction() = 0;

    /**
     * Returns the type converter of the active session
     */
    virtual ITypeConverter& typeConverter() = 0;

  };

}

#endif
