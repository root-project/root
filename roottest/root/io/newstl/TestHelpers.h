#ifndef TEST__HELPER
#define TEST__HELPER

#include "TNamed.h"
#include "TMath.h"
#include "TString.h"

bool IsEquiv(float orig, float copy) {
   float epsilon = 1e-6;
   float diff = orig-copy;
   if (copy < epsilon ) return  TMath::Abs( diff ) < epsilon;
   else return TMath::Abs( diff/copy ) < epsilon;
}

bool IsEquiv(double orig, double copy) {
   double epsilon = 1e-14;
   double diff = orig-copy;
// std::cerr << "epsilon = " << epsilon 
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
   virtual bool IsEquiv(const Helper &rhs) const { return  val==rhs.val && ::IsEquiv(dval,rhs.dval); }
   bool operator<(const Helper &rhs) const { return val<rhs.val; }
   virtual const char* CompMsg(const Helper& copy) const {
      return Form("Helper object wrote %d %g and read %d %g\n",
                  val,dval,copy.val,copy.dval);
   }
};

class HelperClassDef : public Helper {
 public:
   HelperClassDef() {}
   explicit HelperClassDef(int v,double d) : Helper(v,d) {}
   virtual ~HelperClassDef() {};

   virtual const char* CompMsg(const Helper& copy) const {
      return Form("HelperClassDef object wrote %d %g and read %d %g\n",
                  val,dval,copy.val,copy.dval);
   }
   ClassDef(HelperClassDef,1);
};

class HelperDerived : public HelperClassDef {
 public:
   float f;
   HelperDerived() : f(-1) {};
   explicit HelperDerived(int v,double d, float finput) : HelperClassDef(v,d),f(finput) {};
 
   virtual bool IsEquiv(const Helper &rhs) const { 
      bool result = Helper::IsEquiv(rhs);
      if (result) {
         const HelperDerived *drhs = dynamic_cast<const HelperDerived*>(&rhs);
         if (!drhs) return false;
         result =  ::IsEquiv(f,drhs->f);
      }
      return result;
   }

   virtual const char* CompMsg(const Helper& copy) const {
      const HelperDerived *drhs = dynamic_cast<const HelperDerived*>(&copy);
      if (drhs==0) {
         TString msg = Helper::CompMsg(copy);
         return Form("Wrong type (expected %s and got %s) and %s\n",
                     "HelperDerived","THelper",msg.Data());
      }
      return Form("HelperDerived object wrote %d %g %f and read %d %g %f\n",val,dval,f,drhs->val,drhs->dval,drhs->f);
   }

   ClassDef(HelperDerived,1);
};

class THelper : public Helper, public TObject {
public:
   THelper() {};
   explicit THelper(int v,double d) : Helper(v,d) {};
   virtual const char* CompMsg(const Helper& icopy) const {
      const THelper *copy = dynamic_cast<const THelper*>(&icopy);
      if (copy==0) return "Wrong type (expected THelper)\n";
      return Form("THelper object wrote %d %g and read %d %g\n",val,dval,copy->val,copy->dval);
   }
   ClassDef(THelper,1);
};

class THelperDerived : public THelper {
 public:
   float f;
   THelperDerived() : f(-1) {};
   explicit THelperDerived(int v,double d, float finput) : THelper(v,d),f(finput) {};
 
   virtual bool IsEquiv(const Helper &rhs) const { 
      bool result = Helper::IsEquiv(rhs);
      if (result) {
         const THelperDerived *drhs = dynamic_cast<const THelperDerived*>(&rhs);
         if (!drhs) return false;
         result =  ::IsEquiv(f,drhs->f);
      }
      return true;
   }

   virtual const char* CompMsg(const Helper& copy) const {
      const THelperDerived *drhs = dynamic_cast<const THelperDerived*>(&copy);
      if (drhs==0) {
         TString msg = THelper::CompMsg(copy);
         return Form("Wrong type (expected %s and got %s) and %s\n",
                     "THelperDerived","THelper",msg.Data());
      }
      return Form("THelperDerived object wrote %d %g %f and read %d %g %f\n",val,dval,f,drhs->val,drhs->dval,drhs->f);
   }

   ClassDef(THelperDerived,1);
};

template <class T> class GHelper {
public:
   T val;
   bool defaultConstructed;
   GHelper() : val(0),defaultConstructed(true) {}
   explicit GHelper(int v) : val(v),defaultConstructed(false) {}
};

enum EHelper { kZero = 0, kOne, kTwo,
               kHelperEnd = 40 };

bool operator<(const TNamed&lhs, const TNamed&rhs) {
   return strcmp(lhs.GetName(),rhs.GetName()) < 0;
}

template <class T> class PtrCmp {
public:
   bool operator()(const T * lhs, const T * rhs) const {
      if (lhs==0) return (rhs!=0);
      if (rhs==0) return false;
      return *lhs < *rhs;
   }
};

class TList;
void fillListOfDir(TList &l);

#endif // TEST__HELPER
