#ifndef DATABLOCK1_H
#define DATABLOCK1_H
 
#include "DataBlockBase.h"
#include "PureAbstractInterface.h"
 
class DataBlock1 : public DataBlockBase, public PureAbstractInterface {
 
 public:
 
  DataBlock1();
  virtual ~DataBlock1();
 
  virtual Short_t      GetXyzzy() const;
  virtual Short_t      GetAbc()   const;

 private:

  ClassDef(DataBlock1,1)
};

inline Short_t DataBlock1::GetXyzzy() const { return fRawBlock[0]; }
inline Short_t DataBlock1::GetAbc() const { return fRawBlock[1]; }

#endif
