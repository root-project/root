#ifndef RECRECORDIMP_H
#define RECRECORDIMP_H

#include "TObject.h" 

template <class T> 
class RecRecordImp : public TObject {

 public:

   RecRecordImp() {} 
   RecRecordImp(const T& header): fHeader(header) {}
   ~RecRecordImp() override {}

   // State testing methods
   virtual const T& GetHeader() const { return fHeader; }
   void Print(Option_t* option = "") const override;

 private:
   T                  fHeader;       // header base class

   ClassDefOverride(RecRecordImp<T>,1)
};

#endif  
