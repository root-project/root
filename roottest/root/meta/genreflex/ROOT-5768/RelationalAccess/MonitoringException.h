#ifndef RELATIONALACCESS_MONITORING_EXCEPTION_H
#define RELATIONALACCESS_MONITORING_EXCEPTION_H

#include "CoralBase/Exception.h"

namespace coral {

  /**
   * Class MonitoringException
   *
   * Base exception class for the errors related to / produced by an
   * IMonitoring implementation.
   */

  class MonitoringException : public Exception
  {
  public:
    /// Constructor
    MonitoringException( const std::string& message,
                         const std::string& methodName,
                         const std::string& moduleName ) :
      Exception( message, methodName, moduleName )
    {}

    MonitoringException() {}

    /// Destructor
    ~MonitoringException() throw() override {}
  };

}

#endif
