// $Id: HvsPathHandlerException.h,v 1.3 2009-12-17 18:38:53 avalassi Exp $
#ifndef RELATIONALCOOL_HVSPATHHANDLEREXCEPTION_H
#define RELATIONALCOOL_HVSPATHHANDLEREXCEPTION_H 1

// Include files
#include <string>
#include "CoolKernel/Exception.h"

namespace cool
{

  /** @class HvsPathHandlerException
   *
   *  Exception thrown by the HvsPathHandler class.
   */

  class HvsPathHandlerException : public Exception {

  public:

    /// Constructor
    HvsPathHandlerException( const std::string& message )
      : Exception( message, "HvsPathHandler" ) {}

    /// Destructor
    virtual ~HvsPathHandlerException() throw() {}

  };

}

#endif // RELATIONALCOOL_HVSPATHHANDLEREXCEPTION_H
