#ifndef CORAL_CORALBASE_DATE_H
#define CORAL_CORALBASE_DATE_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Include files
#include "CoralBase/boost_datetime_headers.h"

namespace coral
{
  /**
   * @class Date Date.h CoralBase/Date.h
   *
   * A class defining the ANSI DATE type implemented
   * as the boost::posix_time::ptime class
   */
  class Date
  {
  public:

    /// Default constructor
    Date();

    /// Constructor
    Date( int year, int month, int day );

    /// Destructor
    ~Date();

    /// Constructor from a posix time
    explicit Date( const boost::posix_time::ptime& );

    /// Copy constructor
    Date( const Date& rhs );

    /// Assignment operator
    Date& operator=( const Date& rhs );

    /// Equal operator
    bool operator==( const Date& rhs ) const;

    /// Comparison operator
    bool operator!=( const Date& rhs ) const;

    /// Returns the year
    int year() const;

    /// Returns the month [1-12]
    int month() const;

    /// Returns the day [1-31]
    int day() const;

    /// Returns the underlying object
    const boost::posix_time::ptime& time() const;

  public:

    static Date today();

  private:

    /// The actual time object
    boost::posix_time::ptime m_time;

  };

}


/// Inline methods
inline
coral::Date::Date( const boost::posix_time::ptime& pT )
  : m_time( pT )
{
}


inline
coral::Date::Date( const coral::Date& rhs )
  : m_time( rhs.m_time )
{
}


inline coral::Date&
coral::Date::operator=( const coral::Date& rhs )
{
#ifdef CORAL240CO
  if ( this == &rhs ) return *this;  // Fix Coverity SELF_ASSIGN bug #95355
#endif
  m_time = rhs.m_time;
  return *this;
}


inline int
coral::Date::year() const
{
  return m_time.date().year();
}


inline int
coral::Date::month() const
{
  return m_time.date().month();
}


inline int
coral::Date::day() const
{
  return m_time.date().day();
}


inline const boost::posix_time::ptime&
coral::Date::time() const
{
  return m_time;
}


inline bool
coral::Date::operator==( const coral::Date& rhs ) const
{
  return ( this->year() == rhs.year() &&
           this->month() == rhs.month() &&
           this->day() == rhs.day() );
}


inline bool
coral::Date::operator!=( const coral::Date& rhs ) const
{
  return ( this->year() != rhs.year() ||
           this->month() != rhs.month() ||
           this->day() != rhs.day() );
}
#endif
