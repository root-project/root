#include <Riostream.h>
#include "TFile.h"
#include "TRef.h"
#include "TClass.h"
#include "TStreamerInfo.h"

class MyClass : public TObject {
   public:
     Int_t  x;

     Double_t    x1;
     Double_t    x2;
     Double_t    x3;
     Double_t    x4;
     Double_t    x5;
     Double_t    x6;
     Double_t    x7;
     Double_t    x8;
//     Int_t       skipsize;
     Double_t    x9;
//     Float_t  *skip3;   // [skipsize]
     Double_t    x10;
     Int_t       x11;

     MyClass() { /*skip3=0;*/ }

   ClassDef(MyClass,2);
};

void skipRead(bool withxml = 0)
{
   if (withxml) {
      TFile* f = TFile::Open("skiptestfile.xml");
      if (f==0) return;
      cout << "Reading .xml file\n";

      MyClass *m; f->GetObject("abc",m);

      cout << "x1  = " << m->x1 << endl;
      cout << "x2  = " << m->x2 << endl;
      cout << "x3  = " << m->x3 << endl;
      cout << "x4  = " << m->x4 << endl;
      cout << "x5  = " << m->x5 << endl;
      cout << "x6  = " << m->x6 << endl;
      cout << "x7  = " << m->x7 << endl;
      cout << "x8  = " << m->x8 << endl;
      cout << "x9  = " << m->x9 << endl;
      cout << "x10 = " << m->x10 << endl;
      cout << "x11 = " << m->x11 << endl;

      MyClass::Class()->GetStreamerInfo(1)->ls("noaddr");

      delete m;
      delete f;
   }
   cout << "Reading .root file\n";
   TFile* f = TFile::Open("skiptestfile.root");
   MyClass *m; f->GetObject("abc",m);

   cout << "x1  = " << m->x1 << endl;
   cout << "x2  = " << m->x2 << endl;
   cout << "x3  = " << m->x3 << endl;
   cout << "x4  = " << m->x4 << endl;
   cout << "x5  = " << m->x5 << endl;
   cout << "x6  = " << m->x6 << endl;
   cout << "x7  = " << m->x7 << endl;
   cout << "x8  = " << m->x8 << endl;
   cout << "x9  = " << m->x9 << endl;
   cout << "x10 = " << m->x10 << endl;
   cout << "x11 = " << m->x11 << endl;

   MyClass::Class()->GetStreamerInfo(1)->ls("noaddr");

   delete m;
   delete f;
}

