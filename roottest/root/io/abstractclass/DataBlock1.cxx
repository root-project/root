#include "DataBlock1.h"

ClassImp(DataBlock1)

DataBlock1::DataBlock1()
  : DataBlockBase()
{
  // default ctor
  for (int i=2; i<fSize; ++i) fRawBlock[i] = 111;
}

DataBlock1::~DataBlock1()
{
  // do nothing special
}
