#ifndef TEST__HELPER
#define TEST__HELPER

#include "TObject.h"

bool IsEquiv(float orig, float copy) {
   float epsilon = 1e-6;
   float diff = orig-copy;
   if (copy < epsilon ) return  TMath::Abs( diff ) < epsilon;
   else return TMath::Abs( diff/copy ) < epsilon;
}

bool IsEquiv(double orig, double copy) {
   double epsilon = 1e-14;
   double diff = orig-copy;
//    std::cerr << "epsilon = " << epsilon 
//              << " diff = " << diff 
//              << " div  = " << diff/copy
//              << " abs = " << TMath::Abs( diff/copy )
//              << " bool = " << (TMath::Abs( diff/copy ) < epsilon) << std::endl;
   if (copy < epsilon ) return  TMath::Abs( diff ) < epsilon;
   else return TMath::Abs( diff/copy ) < epsilon;
}

class nonvirtHelper {
public:
   unsigned int val;
   double dval;
   nonvirtHelper() : val(0),dval(0) {}
   explicit nonvirtHelper(int v,double d) : val(v),dval(d) {}
   ~nonvirtHelper() {};

   bool IsEquiv(const nonvirtHelper &rhs) const { return (val==rhs.val) && ::IsEquiv(dval,rhs.dval); }
   bool operator<(const nonvirtHelper &rhs) const { return val<rhs.val; }
};

class Helper {
public:
   unsigned int val;
   double dval;   
   Helper() : val(0),dval(0) {}
   explicit Helper(int v,double d) : val(v),dval(d) {}
   virtual ~Helper() {};
   //bool operator==(const Helper &rhs) const { return val==rhs.val; }
   //bool operator!=(const Helper &rhs) const { return !(*this==rhs); }
   bool IsEquiv(const Helper &rhs) const { return  val==rhs.val && ::IsEquiv(dval,rhs.dval); }
   bool operator<(const Helper &rhs) const { return val<rhs.val; }
};

class THelper : public Helper, public TObject {
public:
   THelper() {};
   explicit THelper(int v,double d) : Helper(v,d) {};
   ClassDef(THelper,1);
};

template <class T> class GHelper {
public:
   T val;
   bool defaultConstructed;
   GHelper() : val(0),defaultConstructed(true) {}
   explicit GHelper(int v) : val(v),defaultConstructed(false) {}
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
