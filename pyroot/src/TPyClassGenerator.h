// Author: Wim Lavrijsen   May 2004

#ifndef ROOT_TPyClassGenerator
#define ROOT_TPyClassGenerator

// ROOT
#ifndef ROOT_TClassGenerator
#include "TClassGenerator.h"
#endif


class TPyClassGenerator : public TClassGenerator {
public:
   virtual TClass* GetClass( const char* classname, Bool_t load );
   virtual TClass* GetClass( const type_info& typeinfo, Bool_t load );
};

#endif // !ROOT_TPyClassGenerator
