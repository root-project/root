#ifndef TEST__HELPER
#define TEST__HELPER

#include "TObject.h"

class Helper {
public:
   unsigned int val;
   Helper() : val(0) {}
   explicit Helper(int v) : val(v) {}
   virtual ~Helper() {};
   //bool operator==(const Helper &rhs) const { return val==rhs.val; }
   //bool operator!=(const Helper &rhs) const { return !(*this==rhs); }
   bool IsEquiv(const Helper &rhs) const { return  val==rhs.val; }
};

class THelper : public Helper, public TObject {
public:
   THelper() {};
   explicit THelper(int v) : Helper(v) {};
   ClassDef(THelper,1);
};

enum EHelper { kZero = 0, kOne, kTwo,
               kEnd = 40 };

#endif // TEST__HELPER
