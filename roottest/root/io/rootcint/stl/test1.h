
#ifndef ROOT_TFVirtualIO
#define ROOT_TFVirtualIO

#include <map>
#include "TNamed.h"

class TFVirtualIO
{
public:
 virtual  void GetColNames(std::map<TString, TNamed> &columns) = 0;
 ClassDef(TFVirtualIO,0) // interface definition to files storing TFIOElements
};

#endif

