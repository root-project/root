#include "DataBlockBase.h"
#include <iostream>
#include <iomanip>
using namespace std;

ClassImp(DataBlockBase)

DataBlockBase::DataBlockBase()
  : fSize(10)
{
  if (fSize > 0) {
    fRawBlock = new Int_t [fSize];
    for (int i=0; i<fSize; ++i) fRawBlock[i] = i;
  }
}

DataBlockBase::~DataBlockBase()
{
  if (fSize > 0) delete [] fRawBlock;
}


void DataBlockBase::Print(Option_t * /* option */ ) const
{
  cout << this->GetName();
  for (int i=0; i<fSize; ++i) cout << " " << fRawBlock[i];
  cout << endl;
}
