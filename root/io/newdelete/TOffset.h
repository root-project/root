// @(#)root/base:$Name:  $:$Id: TOffset.h,v 1.1.1.1 2002/08/02 22:25:36 pcanal Exp $
// Author: Victor Perev   08/05/02

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
//#pragma link C++ class TOffset;
#endif

#ifndef ROOT_TOffset
#define ROOT_TOffset



#include "Rtypes.h"

class TList;
class TObject;
class TClass;


class TOffset {

public:
   TOffset(TClass *cl,Int_t all=0);
  ~TOffset();
};
#ifdef __CINT__
#pragma link C++ class TOffset+;
#endif
#endif
