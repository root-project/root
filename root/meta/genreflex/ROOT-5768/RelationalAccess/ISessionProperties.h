#ifndef RELATIONALACCESS_ISESSIONPROPERTIES_H
#define RELATIONALACCESS_ISESSIONPROPERTIES_H

#include <string>

namespace coral {

  /**
   * Class ISessionProperties
   *
   * Interface providing session info to the related ISessionProxy.
   */
  class ISessionProperties
  {
  public:
    /**
     * Virtual destructor
     */
    virtual ~ISessionProperties() {}

    /**
     * Returns the name of the RDBMS flavour..
     */
    virtual std::string flavorName() const = 0;

    /**
     * Returns the version of the database server.
     */
    virtual std::string serverVersion() const = 0;

  };
}

#endif
