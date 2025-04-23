#ifndef CONNECTIONSERVICE_WEBCACHECONTROL_EXCEPTION_H
#define CONNECTIONSERVICE_WEBCACHECONTROL_EXCEPTION_H 1

#include "CoralBase/Exception.h"

namespace coral {

  /**
   * Class WebCacheControlException
   *
   * Base exception class for the errors related to / produced by an
   * IWebCacheControl implementation.
   */
  class WebCacheControlException : public Exception
  {
  public:
    /// Constructor
    WebCacheControlException( const std::string& message,
                              const std::string& methodName,
                              const std::string& moduleName  ) :
      Exception( message, methodName, moduleName )
    {}

    WebCacheControlException() {}

    /// Destructor
    ~WebCacheControlException() throw() override {}

  };


}

#endif
