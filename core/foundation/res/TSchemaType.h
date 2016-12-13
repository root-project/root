// @(#)root/core:$Id$

#ifndef R__TSCHEMATYPE_H
#define R__TSCHEMATYPE_H

// NOTE: #included by libCore and libCling. All symbols must be inline.

#include <string>

namespace ROOT {
namespace Internal {
   struct TSchemaType {
      TSchemaType() = default;
      TSchemaType(const char *type, const char *dim) : fType(type),fDimensions(dim) {}
      TSchemaType(const std::string &type, const std::string &dim) : fType(type),fDimensions(dim) {}
      std::string fType;
      std::string fDimensions;
   };
}
}
#endif // R__TSCHEMATYPE_H
