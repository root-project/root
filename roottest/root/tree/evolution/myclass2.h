// myclass.h 
#include "TNamed.h" 
#include "TClass.h" 
class myclass: public TNamed 
{ 
public: 
   Int_t     n; 
   Int_t     eSize; // ! allocated memory size
   Double_t *e;     // [n]
   Int_t     fSize; // ! allocated memory size
   Double_t *f;     // [n]

   myclass()
   :n(0)
   ,eSize(0)
   ,e(0)
   ,fSize(0)
   ,f(0)
   { myclass::Class()->IgnoreTObjectStreamer(); }

   myclass(const char* name)
   :n(0)
   ,eSize(0)
   ,e(0)
   ,fSize(0)
   ,f(0)
   {myclass::Class()->IgnoreTObjectStreamer(); SetName(name);}

   void     Setn(Int_t val) { n = val; }
   Int_t    Getn() { return n; }

   void     Sete(Double_t* val) { e = val; }
   void     SeteAt(Int_t i, Double_t val) { e[i] = val; }
   Double_t GeteAt(Int_t i) { return e[i]; }

   void     Setf(Double_t* val) { f = val; }
   void     SetfAt(Int_t i, Double_t val) { f[i] = val; }
   Double_t GetfAt(Int_t i) { return f[i]; }

   ~myclass() override {delete [] e; delete [] f;}

   void SeteSize(Int_t number) {
      if (number < 0) return;
      if (number == eSize) {
         return;
      }
      Double_t *tmp = e;
      if (number != 0) {
         e = new Double_t[number];
      } else {
         e = 0;
      }
      delete [] tmp;
      eSize = number;
   }

   void SetfSize(Int_t number) {
      if (number < 0) return;
      if (number == fSize) {
         return;
      }
      Double_t *tmp = f;
      if (number != 0) {
         f = new Double_t[number];
      } else {
         f = 0;
      }
      delete [] tmp;
      fSize = number;
   }

   ClassDefOverride(myclass,2)
}; 

#ifdef __MAKECINT__ 
#pragma link C++ class myclass+; 
#endif
