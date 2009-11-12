#include "missingdict_part1.h"

class Content {
public:
  typedef Value value_type;
  std::multimap<int,Value> fMMap; //!
  std::map<int,Value> fMap; //!
  Transient fCache; //!
  int fValue; //!
};

#include "TClass.h"
#include "TStreamerInfo.h"

void missingdict() {
  TClass::GetClass("Content")->GetStreamerInfo()->ls();
}
