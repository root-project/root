#ifndef DATABLOCK2_H
#define DATABLOCK2_H
 
#include "DataBlockBase.h"
#include "PureAbstractInterface.h"
 
class DataBlock2 : public DataBlockBase, public PureAbstractInterface {
 
 public:
 
  DataBlock2();
  virtual ~DataBlock2();
 
  Short_t      GetXyzzy() const override;
  Short_t      GetAbc()   const override;

 private:

  ClassDefOverride(DataBlock2,1)
};

inline Short_t DataBlock2::GetXyzzy() const { return fRawBlock[0]; }
inline Short_t DataBlock2::GetAbc() const { return fRawBlock[1]; }

#endif
