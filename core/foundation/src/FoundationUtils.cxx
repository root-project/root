/// \file FoundationUtils.hxx
///
/// \brief The file contains utilities which are foundational and could be used
/// across the core component of ROOT.
///
///
/// \author Vassil Vassilev <vvasilev@cern.ch>
///
/// \date June, 2019
///
/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/FoundationUtils.hxx>

#include <algorithm>

#include <errno.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif // _WIN32

namespace ROOT {
namespace FoundationUtils {
std::string GetCurrentDir()
{
   char fixedLength[1024];
   char *currWorkDir = fixedLength;
   size_t len = 1024;
   char *result = currWorkDir;

   do {
      if (result == 0) {
         len = 2 * len;
         if (fixedLength != currWorkDir) {
            delete[] currWorkDir;
         }
         currWorkDir = new char[len];
      }
#ifdef WIN32
      result = ::_getcwd(currWorkDir, len);
#else
      result = getcwd(currWorkDir, len);
#endif
   } while (result == 0 && errno == ERANGE);

   std::string output = currWorkDir;
   output += '/';
#ifdef WIN32
   // convert backslashes into forward slashes
   std::replace(output.begin(), output.end(), '\\', '/');
#endif

   if (fixedLength != currWorkDir) {
      delete[] currWorkDir;
   }
   return output;
}

std::string MakePathRelative(const std::string &path, const std::string &base, bool isBuildingROOT /* = false*/)
{
   std::string result(path);

   const char *currWorkDir = base.c_str();
   size_t lenCurrWorkDir = strlen(currWorkDir);
   if (result.substr(0, lenCurrWorkDir) == currWorkDir) {
      // Convert to path relative to $PWD.
      // If that's not what the caller wants, she should pass -I to rootcling and a
      // different relative path to the header files.
      result.erase(0, lenCurrWorkDir);
   }
   // FIXME: This is not a generic approach for an interface. We should rework
   // this part.
   if (isBuildingROOT) {
      // For ROOT, convert module directories like core/base/inc/ to include/
      int posInc = result.find("/inc/");
      if (posInc != -1) {
         result = /*std::string("include") +*/ result.substr(posInc + 5, -1);
      }
   }
   return result;
}
} // namespace FoundationUtils
} // namespace ROOT
