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
   bool operator<(const Helper &rhs) const { return val<rhs.val; }
};

class THelper : public Helper, public TObject {
public:
   THelper() {};
   explicit THelper(int v) : Helper(v) {};
   ClassDef(THelper,1);
};

enum EHelper { kZero = 0, kOne, kTwo,
               kEnd = 40 };

bool operator<(const TNamed&lhs, const TNamed&rhs) {
   return strcmp(lhs.GetName(),rhs.GetName()) < 0;
}

template <class T> class PtrCmp {
public:
   bool operator()(const T * lhs, const T * rhs) {
      if (lhs==0) return (rhs!=0);
      if (rhs==0) return false;
      return *lhs < *rhs;
   }
};

#endif // TEST__HELPER
