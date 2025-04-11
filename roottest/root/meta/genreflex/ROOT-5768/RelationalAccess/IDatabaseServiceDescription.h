#ifndef RELATIONALACCESS_IDATABASESERVICEDESCRIPTION_H
#define RELATIONALACCESS_IDATABASESERVICEDESCRIPTION_H

#include <string>
#include "AccessMode.h"

namespace coral {

  /**
   * Class IDatabaseServiceDescription
   *
   * Interface for the description of a logical database service.
   */
  class IDatabaseServiceDescription {
  public:
    /**
     * Name of the parameter for the connection retrial period
     */
    static std::string connectionRetrialPeriodParam();

    /**
     * Name of the parameter for the connection retrial time out
     */
    static std::string connectionRetrialTimeOutParam();

    /**
     * Name of the parameter for the connection time out
     */
    static std::string connectionTimeOutParam();

    /**
     * Name of the parameter for the exclusion time for the missing connections
     */
    static std::string missingConnectionExclusionTimeParam();

    /**
     * Name of the server name parameter
     */
    static std::string serverNameParam();

    /**
     * Name of the server status parameter
     */
    static std::string serverStatusParam();

    /**
     * Online server status
     */
    static std::string serverStatusOnline();

    /**
     * Offline server status
     */
    static std::string serverStatusOffline();

    /**
     * Returns the actual connection string for the database service.
     */
    virtual std::string connectionString() const = 0;

    /**
     * Returns the string describing the authentication mechanism.
     */
    virtual std::string authenticationMechanism() const = 0;

    /**
     * Returns the access mode of the database service
     */
    virtual AccessMode accessMode() const = 0;

    /**
     * Returns the specified parameter
     */
    virtual std::string serviceParameter(const std::string& parameterName) const = 0;

  protected:
    /// Protected empty destructor
    virtual ~IDatabaseServiceDescription() {}
  };

}

#endif
