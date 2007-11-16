#ifndef FLOAT16_H
#define FLOAT16_H

#include "TFile.h"
#include "TMath.h"
#include "Riostream.h"

class myclass {
public:
#if defined(CASE_double32)
   Double32_t b; 
   Double32_t c;
   Double32_t s;
   Double32_t i;
   Double32_t l;
   Double32_t ll;
   Double32_t uc;
   Double32_t us;
   Double32_t ui;
   Double32_t ul;
   Double32_t ull;
   Double32_t f;
   Double32_t d;
#elif defined(CASE_double32enough)
   Double32_t b;   //[0,20,10]
   Double32_t c;   //[40,20,10]
   Double32_t s;   //[0,20,10]
   Double32_t i;   //[0,20,10]
   Double32_t l;   //[0,20,10]
   Double32_t ll;  //[0,20,10]
   Double32_t uc;  //[0,20,10]
   Double32_t us;  //[0,20,10]
   Double32_t ui;  //[0,20,10]
   Double32_t ul;  //[0,20,10]
   Double32_t ull; //[0,20,10]
   Double32_t f;   //[0,20,10]
   Double32_t d;   //[0,20,10]
#elif defined(CASE_double32tooshort)
   Double32_t b;   //[0,20,2]
   Double32_t c;   //[40,20,2]
   Double32_t s;   //[0,20,2]
   Double32_t i;   //[0,20,2]
   Double32_t l;   //[0,20,2]
   Double32_t ll;  //[0,20,2]
   Double32_t uc;  //[0,20,2]
   Double32_t us;  //[0,20,2]
   Double32_t ui;  //[0,20,2]
   Double32_t ul;  //[0,20,2]
   Double32_t ull; //[0,20,2]
   Double32_t f;   //[0,20,2]
   Double32_t d;   //[0,20,2]
#elif defined(CASE_double32mantis)
   Double32_t b;   //[0,0,3]
   Double32_t c;   //[0,0,5]
   Double32_t s;   //[0,0,3]
   Double32_t i;   //[0,0,3]
   Double32_t l;   //[0,0,3]
   Double32_t ll;  //[0,0,3]
   Double32_t uc;  //[0,0,3]
   Double32_t us;  //[0,0,3]
   Double32_t ui;  //[0,0,3]
   Double32_t ul;  //[0,0,3]
   Double32_t ull; //[0,0,3]
   Double32_t f;   //[0,0,3]
   Double32_t d;   //[0,0,3]
#elif defined(CASE_float16)
   Float16_t b; 
   Float16_t c;
   Float16_t s;
   Float16_t i;
   Float16_t l;
   Float16_t ll;
   Float16_t uc;
   Float16_t us;
   Float16_t ui;
   Float16_t ul;
   Float16_t ull;
   Float16_t f;
   Float16_t d;
#elif defined(CASE_float16enough)
   Float16_t b;   //[0,20,10]
   Float16_t c;   //[40,20,10]
   Float16_t s;   //[0,20,10]
   Float16_t i;   //[0,20,10]
   Float16_t l;   //[0,20,10]
   Float16_t ll;  //[0,20,20]
   Float16_t uc;  //[0,20,10]
   Float16_t us;  //[0,20,10]
   Float16_t ui;  //[0,20,10]
   Float16_t ul;  //[0,20,10]
   Float16_t ull; //[0,20,10]
   Float16_t f;   //[0,20,10]
   Float16_t d;   //[0,20,10]
#elif defined(CASE_float16tooshort)
   Float16_t b;   //[0,20,2]
   Float16_t c;   //[40,20,2]
   Float16_t s;   //[0,20,2]
   Float16_t i;   //[0,20,2]
   Float16_t l;   //[0,20,2]
   Float16_t ll;  //[0,20,2]
   Float16_t uc;  //[0,20,2]
   Float16_t us;  //[0,20,2]
   Float16_t ui;  //[0,20,2]
   Float16_t ul;  //[0,20,2]
   Float16_t ull; //[0,20,2]
   Float16_t f;   //[0,20,2]
   Float16_t d;   //[0,20,2]
#elif defined(CASE_float16mantis)
   Float16_t b;   //[0,0,3]
   Float16_t c;   //[0,0,5]
   Float16_t s;   //[0,0,3]
   Float16_t i;   //[0,0,3]
   Float16_t l;   //[0,0,3]
   Float16_t ll;  //[0,0,3]
   Float16_t uc;  //[0,0,3]
   Float16_t us;  //[0,0,3]
   Float16_t ui;  //[0,0,3]
   Float16_t ul;  //[0,0,3]
   Float16_t ull; //[0,0,3]
   Float16_t f;   //[0,0,3]
   Float16_t d;   //[0,0,3]
#elif defined(CASE_regular)
   bool b; 
   char c;
   short s;
   int i;
   long l;
   long long ll;
   unsigned char uc;
   unsigned short us;
   unsigned int ui;
   unsigned long ul;
   unsigned long long ull;
   float f;
   double d;
#else
   MYTYPE b;
   MYTYPE c;
   MYTYPE s;
   MYTYPE i;
   MYTYPE l;
   MYTYPE ll;
   MYTYPE uc;
   MYTYPE us;
   MYTYPE ui;
   MYTYPE ul;
   MYTYPE ull;
   MYTYPE f;
   MYTYPE d;
#endif
   myclass() : b(false),c('0'),s(0),i(0),l(0),ll(0),uc('0'),us(0),ui(0),ul(0),ull(0),f(0),d(0) {}

   void Seed() {
      b = true;
      c = '2';
      s = 3;
      i = 4;
      l = 5;
      ll = 6;
      uc = 7;
      us = 8;
      ui = 9;
      ul = 10;
      ull = 11;
      f = 12;
      d = 13;
   }
};

template <typename T>
T diff(const T& left, const T& right)
{
   if (left > right) {
      return left - right;
   } else {
      return right - left;
   }
}

template <typename T> 
bool Compare(const T& ref, const T&obj, const char *name) {
   bool result = diff(ref,obj) <= 1;
   if (!result) {
      cout << "Value of " << name << " is " << obj << " instead of " << ref << endl;
   }
   return result;
}

bool Compare(myclass *ref, myclass *obj) {
   bool result = true;
   result = Compare(ref->b,obj->b,"b") && result;
   result = Compare(ref->c,obj->c,"c") && result;
   result = Compare(ref->s,obj->s,"s") && result;
   result = Compare(ref->i,obj->i,"i") && result;
   result = Compare(ref->l,obj->l,"l") && result;
   result = Compare(ref->ll,obj->ll,"ll") && result;
   result = Compare(ref->uc,obj->uc,"uc") && result;
   result = Compare(ref->us,obj->us,"us") && result;
   result = Compare(ref->ui,obj->ui,"ui") && result;
   result = Compare(ref->ul,obj->ul,"ul") && result;
   result = Compare(ref->ull,obj->ull,"ull") && result;
   result = Compare(ref->f,obj->f,"f") && result;
   result = Compare(ref->d,obj->d,"d") && result;
   return result;
}

void write(const char *filename)
{
   TFile *f = TFile::Open(filename,"RECREATE");
   myclass *c = new myclass;
   c->Seed();
   f->WriteObject(c,"myobj");
   f->Write();
}

bool readfile(const char *filename, Bool_t checkValue = kTRUE)
{
   TFile *f = TFile::Open(filename,"READ");
   if (f==0) return false;

   cout << "Reading " << filename << endl;
   myclass *c; f->GetObject("myobj",c);
   myclass *ref = new myclass;
   ref->Seed();

   if (checkValue) {
      return Compare(ref,c);
   } else {
      return c != 0;
   }
}

#ifdef __MAKECINT__
#pragma link C++ class myclass+;
#pragma link C++ function readfile;
#pragma link C++ function write;
#endif

#endif
