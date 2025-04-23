#ifndef CANDSTRIPHANDLE_H
#define CANDSTRIPHANDLE_H

#include "TObject.h"

class CandStripHandle : public TObject
{

public:
  CandStripHandle();
  virtual ~CandStripHandle();

ClassDef(CandStripHandle,1)           // User access handle to CandStrip
};

#endif                                              // CANDSTRIPHANDLE_H
