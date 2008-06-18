// @(#)root/memstat:$Name$:$Id$
// Author: M.Ivanov -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma extra_include "vector";
#include <vector>

#pragma link C++ typedef UIntVector_t;
#pragma link C++ typedef IntVector_t;

#pragma link C++ class TInfoStamp+;
#pragma link C++ class TCodeInfo+;
#pragma link C++ class TStackInfo+;
#pragma link C++ class TMemStatManager+;
#pragma link C++ class TMemStat;
#pragma link C++ class TMemStatDepend;

#pragma link C++ namespace Memstat; 

#pragma link C++ function Memstat::dig2bytes(Long64_t);

#endif
