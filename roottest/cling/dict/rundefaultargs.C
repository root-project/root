#include <iostream>
#include "TROOT.h"
#include "TClass.h"
#include "TList.h"
#include "TMethod.h"

using namespace std;

class TDefaultArgs 
{
public:
   TDefaultArgs() 
   { 
      Xint();
      Xfloat();
      Xchar();
      Xstring();
      Xstdstring();
      Xenum();
      XcharFancy();
      XcharFancy2();
      XcharFancy3();
      XFunc();
      XFunc2();
      cout << endl;
   }
   enum Eenum {
      kEnumConst=16
   };
   
   static const char* GetDefArg(const char* da) 
   {
      return da;
   }
   
   virtual ~TDefaultArgs() {}
   
private:

   void Xint(int i=
             -1) 
   {
      cout << i << endl;
   }
   void Xfloat(float f
               =1.01) 
   {
      cout << f << endl;
   }
   void Xchar(char c='a') 
   {
      cout << c << endl;
   }
   void Xstring(const char* s="abc:,def)ghi'jkl\"mno'pqr<stu") 
   {
      cout << s << endl;
   }
   void Xstdstring(const std::string s = "a std string arg's conversion")
   {
      cout << s.c_str() << endl;
   }
   void Xenum(const Eenum e 
              =  TDefaultArgs::kEnumConst) 
   {
      cout << e << endl;
   }
   void XcharFancy(const char c = '\'') 
   {
      cout << c << endl;
   }
   void XcharFancy2(const char c = '"') 
   {
      cout << c << endl;
   }
   void XcharFancy3(const char c = /* */' ') 
   {
      cout << c << endl;
   }
   void XFunc(const char* s = GetDefArg("an arg's complexity")) 
   {
      cout << s << endl;
   }
   void XFunc2(const char* s = GetDefArg("'nested \"\"' quotes")) 
   {
      cout << s << endl;
   }

   ClassDef(TDefaultArgs,0);
};

ClassImp(TDefaultArgs)

void rundefaultargs() 
{
   TDefaultArgs da;
   TIter iFunc(TClass::GetClass("TDefaultArgs")->GetListOfMethods());
   TMethod* func = 0;
   while ((func = (TMethod*)iFunc()))
      if (func->GetName()[0] == 'X')
         cout << func->GetPrototype() << endl;
}

