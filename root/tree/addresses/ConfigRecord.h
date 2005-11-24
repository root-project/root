
#include "RecDataRecord.h"
#include "RecHeader.h"

class ConfigRecord : public RecDataRecord<RecHeader>
{

public:
  ConfigRecord() {}
  ConfigRecord(const RecHeader& header) : RecDataRecord<RecHeader>(header){} 
  virtual ~ConfigRecord() {}


ClassDef(ConfigRecord,1)                                 // ConfigRecord
};
