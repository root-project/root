// myclass.h 
#include "TNamed.h" 
#include "TClass.h" 
class myclass: public TNamed 
{ 
public: 
   Int_t     n; 
   Int_t     eSize; // ! allocated memory size
   Double_t *e;     // [n]
   Int_t     gSize; // ! allocated memory size
   Double_t *g;     // [n]

   myclass()
   :n(0)
   ,eSize(0)
   ,e(0)
   ,gSize(0)
   ,g(0)
   {myclass::Class()->IgnoreTObjectStreamer();}

   myclass(const char* name)
   :n(0)
   ,eSize(0)
   ,e(0)
   ,gSize(0)
   ,g(0)
   {myclass::Class()->IgnoreTObjectStreamer();SetName(name);}

   void     Setn(Int_t val) { n = val; }
   Int_t    Getn() { return n; }

   void     Sete(Double_t* val) { e = val; }
   void     SeteAt(Int_t i, Double_t val) { e[i] = val; }
   Double_t GeteAt(Int_t i) { return e[i]; }

   void     Setg(Double_t* val) { g = val; }
   void     SetgAt(Int_t i, Double_t val) { g[i] = val; }
   Double_t GetgAt(Int_t i) { return g[i]; }

   ~myclass() override { delete [] e; delete [] g; }

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

   void SetgSize(Int_t number) {
      if (number < 0) return;
      if (number == gSize) {
         return;
      }
      Double_t *tmp = g;
      if (number != 0) {
         g = new Double_t[number];
      } else {
         g = 0;
      }
      delete [] tmp;
      eSize = number;
   }

   ClassDefOverride(myclass,1)
}; 

#ifdef __MAKECINT__ 
#pragma link C++ class myclass+; 
#endif
