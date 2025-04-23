// $Id: boost_filesystem_headers.h,v 1.2 2013-02-27 10:45:09 avalassi Exp $
#ifndef CORALBASE_BOOSTFILESYSTEMHEADERS_H
#define CORALBASE_BOOSTFILESYSTEMHEADERS_H 1

// Disable warnings triggered by the Boost 1.42.0 headers
// See http://wiki.services.openoffice.org/wiki/Writing_warning-free_code
// See also http://www.artima.com/cppsource/codestandards.html
// See also http://gcc.gnu.org/onlinedocs/gcc-4.1.1/cpp/System-Headers.html
// See also http://gcc.gnu.org/ml/gcc-help/2007-01/msg00172.html
#if defined __GNUC__
#pragma GCC system_header
#endif

// Disable warnings triggered by the Boost 1.50.0 headers on icc (bug #100415)
#if defined __ICC
#pragma warning(disable:522)
#endif

// Include files
#include <boost/filesystem.hpp>

#endif // CORALBASE_BOOSTFILESYSTEMHEADERS_H
