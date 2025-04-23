// $Id: uppercaseString.h,v 1.5 2009-12-16 17:17:38 avalassi Exp $
#ifndef UPPERCASESTRING_H
#define UPPERCASESTRING_H 1

// Include files
#include <algorithm>
#include <cctype>
#include <string>

namespace cool {

  inline const std::string uppercaseString( const std::string& aString ) {
    std::string aStringUp = aString;
    std::transform
      ( aStringUp.begin(), aStringUp.end(), aStringUp.begin(), toupper );
    return aStringUp;
  }

}

#endif // UPPERCASESTRING_H
