#ifndef RECRECORDIMP_H
#define RECRECORDIMP_H

#include "TObject.h" 

template <class T> 
class RecRecordImp : public TObject {

 public:

   RecRecordImp() {} 
   RecRecordImp(const T& header): fHeader(header) {}
   virtual ~RecRecordImp() {}

   // State testing methods
   virtual const T& GetHeader() const { return fHeader; }
   virtual void Print(Option_t* option = "") const;

 private:
   T                  fHeader;       // header base class

   ClassDef(RecRecordImp<T>,1)
};

#endif  
