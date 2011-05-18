// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen   May 2004

#ifndef ROOT_TPyClassGenerator
#define ROOT_TPyClassGenerator

// ROOT
#ifndef ROOT_TClassGenerator
#include "TClassGenerator.h"
#endif


class TPyClassGenerator : public TClassGenerator {
public:
   virtual TClass* GetClass( const char* name, Bool_t load );
   virtual TClass* GetClass( const type_info& typeinfo, Bool_t load );
   virtual TClass* GetClass( const char* name, Bool_t load, Bool_t silent );
   virtual TClass* GetClass( const type_info& typeinfo, Bool_t load, Bool_t silent );
};

#endif // !ROOT_TPyClassGenerator
