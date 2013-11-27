#ifndef CORAL_BASE_EXCEPTION_H
#define CORAL_BASE_EXCEPTION_H

#include <exception>
#include <string>

namespace coral
{

  /// Base exception class for the CORAL system
  class Exception : public std::exception
  {
  public:

    /// Constructor
    Exception( const std::string& message,
               const std::string& methodName,
               const std::string& moduleName );

    /// Constructor with an empty message
    Exception() {};

    /// Destructor
    virtual ~Exception() throw() {}

    /// Set (or reset) the execption message
    void setMessage(const std::string& message);

    /// The error reporting method
    virtual const char* what() const throw() { return m_message.c_str(); }

  private:

    /// The exception message
    std::string m_message;

  };

}
#endif
