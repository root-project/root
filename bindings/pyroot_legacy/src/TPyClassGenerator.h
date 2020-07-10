// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen   May 2004

#ifndef ROOT_TPyClassGenerator
#define ROOT_TPyClassGenerator

// ROOT
#include "TClassGenerator.h"


class TPyClassGenerator : public TClassGenerator {
public:
   TClass *GetClass(const char *name, Bool_t load) override;
   TClass *GetClass(const std::type_info &typeinfo, Bool_t load) override;
   TClass *GetClass(const char *name, Bool_t load, Bool_t silent) override;
   TClass *GetClass(const std::type_info &typeinfo, Bool_t load, Bool_t silent) override;
};

#endif // !ROOT_TPyClassGenerator
