#include <iostream>

using namespace std;

class TDefaultArgs 
{
public:
   TDefaultArgs() 
   { 
      Dint();
      Dfloat();
      Dchar();
      Dstring();
      Dstdstring();
      Denum();
      DcharFancy();
      DcharFancy2();
      DcharFancy3();
      DFunc();
      DFunc2();
      cout << endl;
   }
   enum Eenum {
      kEnumConst=16
   };
   
   static const char* GetDefArg(const char* da) 
   {
      return da;
   }
   
   
private:

   void Dint(int i=
             -1) 
   {
      cout << i << endl;
   }
   void Dfloat(float f
               =0.0001) 
   {
      cout << f << endl;
   }
   void Dchar(char c='a') 
   {
      cout << c << endl;
   }
   void Dstring(const char* s="abc:,def)ghi'jkl\"mno'pqr<stu") 
   {
      cout << s << endl;
   }
   void Dstdstring(const std::string s = "a std string arg's conversion")
   {
      cout << s.c_str() << endl;
   }
   void Denum(const Eenum e 
              =  TDefaultArgs::kEnumConst) 
   {
      cout << e << endl;
   }
   void DcharFancy(const char c = '\'') 
   {
      cout << c << endl;
   }
   void DcharFancy2(const char c = '"') 
   {
      cout << c << endl;
   }
   void DcharFancy3(const char c = /* */' ') 
   {
      cout << c << endl;
   }
   void DFunc(const char* s = GetDefArg("an arg's complexity")) 
   {
      cout << s << endl;
   }
   void DFunc2(const char* s = GetDefArg("'nested \"\"' quotes")) 
   {
      cout << s << endl;
   }
};

void rundefaultargs() 
{
   TDefaultArgs da;
}

