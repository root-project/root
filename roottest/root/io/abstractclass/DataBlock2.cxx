#include "DataBlock2.h"

ClassImp(DataBlock2)

DataBlock2::DataBlock2()
  : DataBlockBase()
{
  // default ctor
  for (int i=2; i<fSize; ++i) fRawBlock[i] = 222;
}

DataBlock2::~DataBlock2()
{
  // do nothing special
}
