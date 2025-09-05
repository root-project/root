#ifndef RECDATARECORD_H
#define RECDATARECORD_H

#include "RecRecordImp.cxx"

template <class T>
class RecDataRecord : public RecRecordImp<T> {

 public:
  // ctors/dtor
  RecDataRecord() {}
  RecDataRecord(const T& header) : RecRecordImp<T>(header) {}
  ~RecDataRecord() override {}

 private:
  // data members

  ClassDefOverride(RecDataRecord,1)

};

#endif

