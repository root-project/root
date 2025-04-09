// $Id: VersionNumber.h,v 1.10 2010-09-16 13:58:53 avalassi Exp $
#ifndef VERSIONNUMBER_H
#define VERSIONNUMBER_H 1

// Include files
#include <string>
#include <sstream>
#include <stdexcept>

// forward declarations
class VersionNumber;
inline std::ostream& operator<< ( std::ostream& str, const VersionNumber& vers );

/** @class VersionNumber VersionNumber.h
 *
 *  Class used to handle version numbers of format "x.y.z".
 *  There is no strict format check (2.8.9 and 2.8.10 are both allowed).
 *
 *  @author Marco Clemencic
 *  @date   2006-11-09
 */
class VersionNumber {

public:

  /// Standard constructor
  inline VersionNumber( const std::string &vers ){ initialize(vers); }

  /// Standard constructor (from char array)
  inline VersionNumber( const char *vers ){ initialize(vers); }

  // accessors

  inline int majorVersion() const { return m_major; }
  inline int minorVersion() const { return m_minor; }
  inline int patchVersion() const { return m_patch; }

  // comparison operators

  inline bool operator== (const VersionNumber& rhs) const
  {
    return
      ( majorVersion() == rhs.majorVersion() )
      && ( minorVersion() == rhs.minorVersion() )
      && ( patchVersion() == rhs.patchVersion() );
  }

  inline bool operator!= (const VersionNumber& rhs) const
  {
    return ! (*this == rhs );
  }

  inline bool operator< (const VersionNumber& rhs) const
  {
    return
      ( majorVersion() < rhs.majorVersion() )
      || ( ( majorVersion() == rhs.majorVersion() )
           && ( minorVersion() < rhs.minorVersion() ) )
      || ( ( majorVersion() == rhs.majorVersion() )
           && ( minorVersion() == rhs.minorVersion() )
           && ( patchVersion() < rhs.patchVersion() ) );
  }

  inline bool operator<= (const VersionNumber& rhs) const
  {
    return *this == rhs || *this < rhs;
  }

  inline bool operator> (const VersionNumber& rhs) const
  {
    return !( *this <= rhs );
  }

  inline bool operator>= (const VersionNumber& rhs) const
  {
    return *this == rhs || *this > rhs;
  }

  /// Conversion to string
  inline operator std::string () const
  {
    std::ostringstream s;
    s << *this;
    return s.str();
  }

private:

  void initialize(const std::string &vers)
  {
    m_major = m_minor = m_patch = -1;

    std::istringstream s(vers);
    char sep;
    s >> m_major >> sep >> m_minor >> sep >> m_patch;
    if ( m_major < 0 || m_minor < 0 || m_patch < 0 ) {
      throw std::runtime_error("Bad version string format: '"+vers+"'");
    }
  }

private:

  int m_major;
  int m_minor;
  int m_patch;

};

inline std::ostream& operator<< ( std::ostream& str, const VersionNumber& vers)
{
  return str << vers.majorVersion()
             << '.' << vers.minorVersion()
             << '.' << vers.patchVersion();
}

#endif // VERSIONNUMBER_H
