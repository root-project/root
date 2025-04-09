
#include "RecHeader.h"
#include "RecDataRecord.cxx"
#include "RecRecordImp.cxx"

class ConfigRecord : public RecDataRecord<RecHeader>
{

public:
  ConfigRecord() {}
  ConfigRecord(const RecHeader& header) : RecDataRecord<RecHeader>(header){} 
  ~ConfigRecord() override {}


ClassDefOverride(ConfigRecord,1)                                 // ConfigRecord
};
