#ifndef DATABLOCKBASE_H
#define DATABLOCKBASE_H
 
#include "TObject.h"
 
class DataBlockBase : public TObject {
 
 public:
 
  DataBlockBase();
  virtual ~DataBlockBase();
 
  inline const Int_t*  GetData() const { return fRawBlock; }
  inline Int_t         GetSize() const { return fSize; }

  virtual void         Print(Option_t *option="") const;

 protected:   // allow derived classes direct access to the data

  Int_t  fSize;      // number of Int_t words
  Int_t *fRawBlock;  //[fSize]

 private:

  ClassDef(DataBlockBase,1)
};

#endif
