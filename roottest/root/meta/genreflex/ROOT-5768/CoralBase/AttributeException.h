// -*- C++ -*-
// $Id: AttributeException.h,v 1.2 2006-02-23 10:47:45 ioannis Exp $
#ifndef CORALBASE_ATTRIBUTE_EXCEPTION_H
#define CORALBASE_ATTRIBUTE_EXCEPTION_H 1

#include "Exception.h"

namespace coral
{
  /// Exception class for the AttributeList-related classes
  class AttributeException : public Exception
  {
  public:
    /// Constructor
    AttributeException( std::string errorMessage = "" ) :
      Exception( errorMessage, "Attribute", "CoralBase" )
    {
    }

    ~AttributeException() throw() override
    {
    }
  };
}

#endif
