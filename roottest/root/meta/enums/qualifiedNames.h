#ifndef __qualifiedNames__
#define __qualifiedNames__

// Here kNone is not used. Instead we go for kNoneNonConflicting to avoid conflicts with
// the name kNone which is in GuiTypes.h

namespace myns{
   enum enpclass{kNoneNonConflicting};
}
enum enpclass{kNoneNonConflicting};

#endif
