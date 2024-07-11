#ifndef CPYCPPYY_TPYCLASSGENERATOR
#define CPYCPPYY_TPYCLASSGENERATOR

// ROOT
#include "TClassGenerator.h"


class TPyClassGenerator : public TClassGenerator {
public:
    virtual TClass* GetClass(const char* name, bool load);
    virtual TClass* GetClass(const std::type_info& typeinfo, bool load);
    virtual TClass* GetClass(const char* name, bool load, bool silent);
    virtual TClass* GetClass(const std::type_info& typeinfo, bool load, bool silent);
};

#endif // !CPYCPPYY_TPYCLASSGENERATOR
