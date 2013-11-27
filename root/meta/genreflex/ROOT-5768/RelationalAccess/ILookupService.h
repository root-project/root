#ifndef RELATIONALACCESS_ILOOKUPSERVICE_H
#define RELATIONALACCESS_ILOOKUPSERVICE_H

#include <string>
#include "AccessMode.h"

namespace coral {

  // forward declarations
  class IDatabaseServiceSet;

  /**
   * Class ILookupService
   *
   * Interface for a database lookup service.
   * Implementations of this interface should be seal services.
   *
   */
  class ILookupService
  {
  public:
    /**
     * Performs a lookup for a given logical connection string.
     *
     * The method creates and returns a new object, the caller of the method is
     * responsible for managing the lifetime of it.
     */
    virtual IDatabaseServiceSet* lookup( const std::string& logicalName,
                                         AccessMode accessMode = Update,
                                         std::string authenticationMechanism = "" ) const = 0;

    /**
     * Sets the input file name
     */
    virtual void setInputFileName(  const std::string& inputFileName ) = 0;

  protected:
    /// Protected empty destructor
    virtual ~ILookupService() {}
  };
}

#endif
