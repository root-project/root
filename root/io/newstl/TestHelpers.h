#ifndef TEST__HELPER
#define TEST__HELPER

#include "TObject.h"

class nonvirtHelper {
public:
   unsigned int val;
   nonvirtHelper() : val(0) {}
   explicit nonvirtHelper(int v) : val(v) {}
   ~nonvirtHelper() {};

   bool IsEquiv(const nonvirtHelper &rhs) const { return  val==rhs.val; }
   bool operator<(const nonvirtHelper &rhs) const { return val<rhs.val; }
};

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

template <class T> class GHelper {
public:
   T val;
   GHelper() : val(0) {}
   explicit GHelper(int v) : val(v) {}
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

class TList;
void fillListOfDir(TList &l);

#endif // TEST__HELPER
