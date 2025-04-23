// $Id: InternalErrorException.h,v 1.3 2009-12-16 17:41:24 avalassi Exp $
#ifndef COOLKERNEL_INTERNALERROREXCEPTION_H
#define COOLKERNEL_INTERNALERROREXCEPTION_H

// Include files
#include "CoolKernel/Exception.h"

namespace cool {

  //--------------------------------------------------------------------------

  /** @class InternalErrorException InternalErrorException.h
   *
   *  Exception thrown when a COOL internal error occurs ("PANIC").
   *  These exceptions signal bugs in the internal logic of COOL algorithms.
   *  **** No such exceptions should ever be thrown. If you catch one,  ***
   *  **** please report this immediately to the COOL development team. ***
   *
   *  @author Andrea Valassi
   *  @date   2008-07-31
   */

  class InternalErrorException : public Exception {

  public:

    /// Constructor
    explicit InternalErrorException( const std::string& message,
                                     const std::string& domain )
      : Exception( "*** COOL INTERNAL ERROR *** " + message, domain ) {}

    /// Destructor
    virtual ~InternalErrorException() throw() {}

  };

  //--------------------------------------------------------------------------

}
#endif
