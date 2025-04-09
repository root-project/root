#ifndef COOLKERNEL_ITIME_H
#define COOLKERNEL_ITIME_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include <ostream>

namespace cool
{

  /** @class ITime ITime.h
   *
   *  Abstract interface to a generic COOL time class.
   *
   *  The implementation details are hidden from the public API.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-17
   */

  class ITime
  {

  public:

    /// Destructor.
    virtual ~ITime() {}

    /// Returns the year.
    virtual int year() const = 0;

    /// Returns the month [1-12].
    virtual int month() const = 0;

    /// Returns the day [1-31].
    virtual int day() const = 0;

    /// Returns the hour [0-23].
    virtual int hour() const = 0;

    /// Returns the minute [0-59].
    virtual int minute() const = 0;

    /// Returns the second [0-59].
    virtual int second() const = 0;

    /// Returns the nanosecond [0-999999999].
    virtual long nanosecond() const = 0;

    /// Print to an output stream.
    virtual std::ostream& print( std::ostream& os ) const = 0;

    /// Comparison operator.
    virtual bool operator==( const ITime& rhs ) const = 0;

    /// Comparison operator.
    virtual bool operator>( const ITime& rhs ) const = 0;

    /// Comparison operator.
    bool operator>=( const ITime& rhs ) const;

    /// Comparison operator.
    bool operator<( const ITime& rhs ) const;

    /// Comparison operator.
    bool operator<=( const ITime& rhs ) const;

    /// Comparison operator.
    bool operator!=( const ITime& rhs ) const;

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    ITime& operator=( const ITime& rhs );
#endif

  };

  /// Print to an output stream.
  std::ostream& operator<<( std::ostream& s, const ITime& time );

  //--------------------------------------------------------------------------

  inline bool ITime::operator>=( const ITime& rhs ) const
  {
    return ( (*this) > rhs || (*this) == rhs );
  }

  //--------------------------------------------------------------------------

  inline bool ITime::operator<( const ITime& rhs ) const
  {
    return !( (*this) >= rhs );
  }

  //--------------------------------------------------------------------------

  inline bool ITime::operator<=( const ITime& rhs ) const
  {
    return !( (*this) > rhs );
  }

  //--------------------------------------------------------------------------

  inline bool ITime::operator!=( const ITime& rhs ) const
  {
    return !( (*this) == rhs );
  }

  //--------------------------------------------------------------------------

  inline std::ostream& operator<<( std::ostream& s, const ITime& time )
  {
    return time.print( s );
  }

  //--------------------------------------------------------------------------

}
#endif // COOLKERNEL_ITIME_H
