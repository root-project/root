#include <memory>
#include <vector>
#include "RefVectorIterator.h"

namespace pat {
   class TriggerObject {
   public:
       TriggerObject();
       int& i() {return i_;} 
   private:
       int i_;
   };
   typedef std::vector<TriggerObject> TriggerObjectCollection;
   typedef edm::RefVectorIterator<TriggerObjectCollection, TriggerObject> TriggerObjectRefVectorIterator;
}

namespace {
  struct dictionary {
    pat::TriggerObjectRefVectorIterator dummy;
  };

}
