// -*- C++ -*-
#ifndef RELATIONALACCESS_RELATIONALSERVICE_EXCEPTION_H
#define RELATIONALACCESS_RELATIONALSERVICE_EXCEPTION_H 1

#include "CoralBase/Exception.h"

namespace coral {

  /**
   * Base class for exceptions related to / thrown by the Relational Service
   */

  class RelationalServiceException : public Exception
  {
  public:
    /// Constructor
    RelationalServiceException( const std::string& message,
                                const std::string& methodName,
                                std::string moduleName = "CORAL/Services/RelationalService" ) :
      Exception( message, methodName, moduleName )
    {}

    RelationalServiceException() {}

    /// Destructor
    ~RelationalServiceException() throw() override {}
  };

  /// Exception thrown when an invalid replica identifier is specified
  class NonExistingDomainException : public RelationalServiceException
  {
  public:
    /// Constructor
    NonExistingDomainException( const std::string& domainName,
                                std::string moduleName = "CORAL/Services/RelationalService",
                                std::string methodName = "IRelationalService::domain" ) :
      RelationalServiceException( "Relational domain \"" + domainName + "\" could not be found",
                                  methodName, moduleName )
    {}

    NonExistingDomainException() {}
    /// Destructor
    ~NonExistingDomainException() throw() override {}
  };

  /// Exception thrown when an invalid operation is requested by the CORAL client
  class InvalidOperationException : public RelationalServiceException
  {
  public:
    /// Constructor
    InvalidOperationException( const std::string& domainName, std::string moduleName = "CORAL/Services/RelationalService", std::string methodName = "IRelationalService::domain" ) :
      RelationalServiceException( "Relational domain \"" + domainName + "\" could not be found", methodName, moduleName )
    {}

    InvalidOperationException() {}

    /// Destructor
    ~InvalidOperationException() throw() override {}
  };

}

#endif
