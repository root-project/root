#ifndef CORAL_BASE_STRING_LIST_H
#define CORAL_BASE_STRING_LIST_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Move StringOps from CoralBase to CoralCommon in CORAL240 (bug #103240)
#ifndef CORAL240SO

#include <vector>
#include <string>

namespace coral
{
  typedef std::vector<std::string> StringList;
}

#endif

#endif // CORAL_BASE_STRING_LIST_H
