#ifndef COOLKERNEL_TIME_H
#define COOLKERNEL_TIME_H

// Include files
#include "CoolKernel/ITime.h"

namespace cool
{

  /** @class Time Time.h
   *
   *  Simple COOL time class.
   *
   *  Basic sanity checks on value ranges are delegated in the internal
   *  implementation to a more complex Time class (eg based on SEAL or BOOST).
   *
   *  @author Andrea Valassi
   *  @date   2007-01-17
   */

  class Time : public ITime
  {

  public:

    /// Destructor.
    virtual ~Time();

    /// Default constructor: returns the current UTC time.
    Time();

    /// Constructor from an explicit date/time - interpreted as a UTC time.
    /// Sanity checks on the ranges of the input arguments are implemented.
    Time( int year,
          int month,
          int day,
          int hour,
          int minute,
          int second,
          long nanosecond );

    /// Copy constructor from another Time.
    Time( const Time& rhs );

    /// Assignment operator from another Time.
    Time& operator=( const Time& rhs );

    /// Copy constructor from any other ITime.
    Time( const ITime& rhs );

    /// Assignment operator from any other ITime.
    Time& operator=( const ITime& rhs );

    /// Returns the year.
    int year() const
    {
      return m_year;
    }

    /// Returns the month [1-12].
    int month() const
    {
      return m_month;
    }

    /// Returns the day [1-31].
    int day() const
    {
      return m_day;
    }

    /// Returns the hour [0-23].
    int hour() const
    {
      return m_hour;
    }

    /// Returns the minute [0-59].
    int minute() const
    {
      return m_minute;
    }

    /// Returns the second [0-59].
    int second() const
    {
      return m_second;
    }

    /// Returns the nanosecond [0-999999999].
    long nanosecond() const
    {
      return m_nanosecond;
    }

    /// Print to an output stream.
    std::ostream& print( std::ostream& os ) const;

    /// Comparison operator.
    bool operator==( const ITime& rhs ) const;

    /// Comparison operator.
    bool operator>( const ITime& rhs ) const;

  private:

    int m_year;
    int m_month;
    int m_day;
    int m_hour;
    int m_minute;
    int m_second;
    long m_nanosecond;

  };

}

#endif // COOLKERNEL_TIME_H
