#ifndef RELATIONALACCESS_LOOKUPSERVICE_EXCEPTION_H
#define RELATIONALACCESS_LOOKUPSERVICE_EXCEPTION_H

#include "CoralBase/Exception.h"

namespace coral {

  /// Base exception class for errors related to the lookup service
  class LookupServiceException : public Exception
  {
  public:
    /// Constructor
    LookupServiceException( const std::string& message,
                            const std::string& methodName,
                            const std::string& moduleName ) :
      Exception( message, methodName, moduleName )
    {}

    LookupServiceException() : Exception() {}

    /// Destructor
    ~LookupServiceException() throw() override {}
  };



  /// Exception thrown when an invalid replica identifier is specified
  class InvalidReplicaIdentifierException : public LookupServiceException
  {
  public:
    /// Constructor
    InvalidReplicaIdentifierException( const std::string& moduleName,
                                       std::string message = "Specified index is out of range",
                                       std::string methodName = "IDatabaseServiceSet::replica" ) :
      LookupServiceException( message, methodName, moduleName )
    {}

    InvalidReplicaIdentifierException() : LookupServiceException() {}

    /// Destructor
    ~InvalidReplicaIdentifierException() throw() override {}
  };

}

#endif
