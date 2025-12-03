#ifndef DATABLOCKBASE_H
#define DATABLOCKBASE_H
 
#include "TObject.h"
 
class DataBlockBase : public TObject {
 
 public:
 
  DataBlockBase();
  virtual ~DataBlockBase();
 
  inline const Int_t*  GetData() const { return fRawBlock; }
  inline Int_t         GetSize() const { return fSize; }

  void         Print(Option_t *option="") const override;

 protected:   // allow derived classes direct access to the data

  Int_t  fSize;      // number of Int_t words
  Int_t *fRawBlock;  //[fSize]

 private:

  ClassDefOverride(DataBlockBase,1)
};

#endif
