#ifndef ConfigRecord_h
#define ConfigRecord_h

#include "RecHeader.h"
#include "RecDataRecord.h"

class ConfigRecord : public RecDataRecord<RecHeader>
{

public:
  ConfigRecord() {}
  ConfigRecord(const RecHeader& header) : RecDataRecord<RecHeader>(header){}
  ~ConfigRecord() override {}


ClassDefOverride(ConfigRecord,1)                                 // ConfigRecord
};

#endif
