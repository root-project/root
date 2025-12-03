#include <Riostream.h>
#include <string>
#include "TFile.h"
#include "TNamed.h"
#include "TCut.h"
#include "TRef.h"
#include "TAttFill.h"

class MyClass : public TObject, public TAttFill {
   public:
     Int_t    x;
     TNamed   obj;
     Int_t    x1;
     TNamed*  pobj;
     Int_t    x2;
     TObject*  skipObjectp;  //->
     Int_t    x3;
     TObject*  skipObjectP;  
     Int_t    x4;
     TCut      skipObject;
     Int_t    x5;
     TRef      skipTRef;
     Int_t    x6;
     TRef      *skipTRefP;
     Int_t    x7;
     TRef      *skipTRefp;   //->
     Int_t    x8;
     Int_t    host1;
     Int_t    host2;
     Int_t    host3;
     Int_t    skipsize;     
     Int_t    x9;
     Float_t  *skip3;   // [skipsize]
     Int_t    x10;
     const char* skipstr;
     Int_t    x11;
     Int_t    x12;
     Int_t    x13;
     
     MyClass() :
       x(0),
       obj("test","testtitle"),
       pobj(0),
       skipObject("CutName","Cut title for test skipObject"),
       skipTRef(),
       skipTRefP(0)
     { 
        skipObjectp = new TCut("CutNamep","Cut title for test skipObjectp");
        skipObjectP = 0;
           
        skipTRefp = new TRef;   

        skipsize = 0; 
        skip3 = 0; 
        skipstr = 0;
     }
     
   ClassDef(MyClass,1);  
};

void skipWrite() {
  MyClass m;
  
  m.x = 100;
  m.x1 = 1001;
  m.x2 = 2002;
  m.x3 = 3003;
  m.x4 = 4004;
  m.x5 = 5005;
  m.x6 = 6006;
  m.x7 = 7007;
  m.x8 = 8008;
  m.x9 = 9009;
  m.x10 = 10010;
  m.x11 = 110011;
  m.x12 = 120012;
  m.x13 = 130013;
  m.pobj = new TNamed("pobj","pobj title");
  m.skipObjectP = new TCut("CutNameP","Cut title for test skipObjectP");
  m.skipTRefP = new TRef;

  m.host1 = 10;
  m.host2 = 20;
  m.host3 = 30;
  
  m.skipsize = 20;  
  m.skip3 = new Float_t[m.skipsize];
  for(int n=0;n<m.skipsize;n++)
    m.skip3[n] = 0;
  m.skip3[5] = 555;
  
  m.skipstr = new char[100];
  strcpy((char*) m.skipstr,"Value of Char Start");

  TFile* f = TFile::Open("skiptestfile.xml","recreate");
  if (f) {
     m.Write("abc");
     delete f;
  }

  f = TFile::Open("skiptestfile.root","recreate");
  m.Write("abc");
  delete f;
}

