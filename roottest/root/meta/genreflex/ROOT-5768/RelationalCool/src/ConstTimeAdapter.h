#ifndef RELATIONALCOOL_CONSTTIMEADAPTER_H
#define RELATIONALCOOL_CONSTTIMEADAPTER_H

// Include files
#include "CoolKernel/ITime.h"

namespace cool
{

  /** @class ConstTimeAdapter ConstTimeAdapter.h
   *
   *  Wrapper of a std::string to the cool::ITime interface.
   *
   *  NB Every method call triggers a string-to-time conversion:
   *  this is because it is assumed that the string may change
   *  during the lifetime of the adapter, hence nothing is cached!
   *
   *  @author Andrea Valassi
   *  @date   2007-03-29
   */

  class ConstTimeAdapter : public ITime
  {

  public:

    /// Destructor.
    virtual ~ConstTimeAdapter();

    /// Constructor from a const std::string reference.
    ConstTimeAdapter( const std::string& time );

    /// Returns the year.
    int year() const;

    /// Returns the month [1-12].
    int month() const;

    /// Returns the day [1-31].
    int day() const;

    /// Returns the hour [0-23].
    int hour() const;

    /// Returns the minute [0-59].
    int minute() const;

    /// Returns the second [0-59].
    int second() const;

    /// Returns the nanosecond [0-999999999].
    long nanosecond() const;

    /// Print to an output stream.
    std::ostream& print( std::ostream& os ) const;

    /// Comparison operator.
    bool operator==( const ITime& rhs ) const;

    /// Comparison operator.
    bool operator>( const ITime& rhs ) const;

  private:

    /// Standard constructor
    ConstTimeAdapter();

    /// Copy constructor from another ConstTimeAdapter.
    ConstTimeAdapter( const ConstTimeAdapter& rhs );

    /// Assignment operator from another ConstTimeAdapter.
    ConstTimeAdapter& operator=( const ConstTimeAdapter& rhs );

  private:

    const std::string& m_time;

  };

}

#endif // COOLKERNEL_TIME_H
