// Mock from cms

namespace cond {

class IOVSequence {
public:

   enum    ScopeType {
            Unknown =-1, Obsolete, Tag, TagInGT,
            ChildTag, ChildTagInGT
            };
   IOVSequence(){};
   ScopeType scope(){return Tag;};
};

}
