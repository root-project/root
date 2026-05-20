/*
  File: roottest/python/function/InstallableFunction.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 07/02/07
  Last: 05/04/15
*/

class FuncLess {
public:
   FuncLess( int i ) : m_int( i ) {}
   int m_int;
};

FuncLess* InstallableFunc( FuncLess* self ) {
   return self;
}

namespace FunctionNS {
   FuncLess* InstallableFunc( FuncLess* self ) {
      return self;
   }
}
