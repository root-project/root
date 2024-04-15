/*
 * Copyright (c) 2023, CERN
 */

#include <Math/IOptions.h>

#include <Math/Error.h>

namespace ROOT {
namespace Math {

double IOptions::RValue(const char *name) const
{
   double val = 0;
   bool ret = GetRealValue(name, val);
   if (!ret)
      MATH_ERROR_MSGVAL("IOptions::RValue", " return 0 - real option not found", name);
   return val;
}

int IOptions::IValue(const char *name) const
{
   int val = 0;
   bool ret = GetIntValue(name, val);
   if (!ret)
      MATH_ERROR_MSGVAL("IOptions::IValue", " return 0 - integer option not found", name);
   return val;
}

std::string IOptions::NamedValue(const char *name) const
{
   std::string val;
   bool ret = GetNamedValue(name, val);
   if (!ret)
      MATH_ERROR_MSGVAL("IOptions::NamedValue", " return empty string - named option not found", name);
   return val;
}

/// method which need to be re-implemented by the derived classes
void IOptions::SetRealValue(const char *, double)
{
   MATH_ERROR_MSG("IOptions::SetRealValue", "Invalid setter method called");
}

void IOptions::SetIntValue(const char *, int)
{
   MATH_ERROR_MSG("IOptions::SetIntValue", "Invalid setter method called");
}

void IOptions::SetNamedValue(const char *, const char *)
{
   MATH_ERROR_MSG("IOptions::SetNamedValue", "Invalid setter method called");
}

/// print options
void IOptions::Print(std::ostream &) const
{
   MATH_INFO_MSG("IOptions::Print", "it is not implemented");
}

} // namespace Math
} // namespace ROOT
