#ifndef RELATIONALACCESS_IRELATIONALSERVICE_H
#define RELATIONALACCESS_IRELATIONALSERVICE_H

#include <string>
#include <vector>

namespace coral {

  // forward declarations
  class IRelationalDomain;

  /**
   * Class IRelationalService
   * Interface class for a relational service which should be implemented as a SEAL service.
   * This service has the responsibility of automatically loading the proper flavour-specific plugins
   * and pass to the client references to the corresponding IRelationalDomain objects.
   * The service is responsible for deciding which implementation to choose for a given RDBMS technology.
   */
  class IRelationalService {
  public:
    /**
     * Lists the currently available technologies
     */
    virtual std::vector< std::string > availableTechnologies() const = 0;

    /**
     * Lists the available implementations for a given technology
     */
    virtual std::vector< std::string > availableImplementationsForTechnology( const std::string& technologyName ) const = 0;

    /**
     * Forces a particular implementation to be used when picking up a technology domain.
     * Returns true in case of success, false if the specified technology or domain are not valid.
     *
     * @param technology The name of the RDBMS technology
     * @param implementation The name of the implementation of the corresponding plugin. If not specified the native implementation is selected
     */
    virtual void setDefaultImplementationForDomain( const std::string& technology,
                                                    std::string implementation = "" ) = 0;

    /**
     * Returns a reference to the underlying IRelationalDomain object given the name of
     * the RDBMS technology.
     * If the corresponding plugin module cannot be found, a NonExistingDomainException is thrown.
     */
    virtual IRelationalDomain& domain( const std::string& flavourName ) = 0;

    /**
     * Returns a reference to the underlying IRelationalDomain object given the full connection
     * string specifying the working schema in a database.
     * If the corresponding plugin module cannot be found, a NonExistingDomainException is thrown.
     */
    virtual IRelationalDomain& domainForConnection( const std::string& connectionString ) = 0;

  protected:
    /// Protected empty destructor
    virtual ~IRelationalService() {}
  };

}

#endif
