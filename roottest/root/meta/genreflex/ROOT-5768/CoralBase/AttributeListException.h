#ifndef CORALBASE_ATTRIBUTELIST_EXCEPTION_H
#define CORALBASE_ATTRIBUTELIST_EXCEPTION_H

#include "Exception.h"

namespace coral {

  /// Exception class for the AttributeList-related classes
  class AttributeListException : public Exception
  {
  public:
    /// Constructor
    AttributeListException( std::string errorMessage = "" ) :
      Exception( errorMessage,
                 "AttributeList",
                 "CoralBase" )
    {}

    ~AttributeListException() throw() override {}
  };

}

#endif
