// $Id: RecordSelectionException.h,v 1.4 2009-12-16 17:41:24 avalassi Exp $
#ifndef COOLKERNEL_RECORDSELECTIONEXCEPTION_H
#define COOLKERNEL_RECORDSELECTIONEXCEPTION_H 1

// Include files
#include <sstream>
#include "CoolKernel/Exception.h"

namespace cool
{

  //--------------------------------------------------------------------------

  /** @class RecordSelectionException
   *
   *  Base class for all exceptions thrown while defining record selections.
   *
   *  @author Martin Wache and Andrea Valassi
   *  @date   2008-07-31
   */

  class RecordSelectionException : public Exception
  {

  public:

    /// Constructor
    explicit RecordSelectionException( const std::string& message,
                                       const std::string& domain )
      : Exception( message, domain ) {}

    /// Destructor
    virtual ~RecordSelectionException() throw() {}

  };

  //--------------------------------------------------------------------------

}

#endif // COOLKERNEL_RECORDSELECTIONEXCEPTION_H
