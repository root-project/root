// @(#)root/pyroot:$Name:  $:$Id: TPyClassGenerator.h,v 1.68 2005/01/28 05:45:41 brun Exp $
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
};

#endif // !ROOT_TPyClassGenerator
