#include <vector>
#include <map>

#include "TObject.h"
using namespace std;

namespace HepExp {
  class Inner { 
  public:
     Inner(int i = 0) : fX(i),fY(i) {}
     int fX; 
     float fY; 
  };
  typedef Inner Alias;
  class OtherInner {
  public:
     OtherInner(int i = 0) : fX(i),fY(i) {}
     int fX;
     float fY;
  };
  class Outer {
  public:
     Outer(int seed = 0) {
        fAlias.push_back(Inner(1 *seed));
        fValue.push_back(OtherInner(2 *seed));
        fMap[0] = OtherInner(3 *seed);
        fMapAlias[0] = Inner(4 *seed);
     }
     vector<OtherInner> fValue;
     vector<Alias> fAlias;
     map<int,OtherInner> fMap;
     map<int,Alias> fMapAlias;

     void Print() {
        fprintf(stdout,"fAlias[0]    : %d %3.2f\n", fAlias[0].fX, fAlias[0].fY);
        fprintf(stdout,"fValue[0]    : %d %3.2f\n", fValue[0].fX, fValue[0].fY);
        fprintf(stdout,"fMap[0]      : %d %3.2f\n", fMap[0].fX, fMap[0].fY);
        fprintf(stdout,"fMapAlias[0] : %d %3.2f\n", fMapAlias[0].fX, fMapAlias[0].fY);
     }
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,99,0)
     ClassDef(Outer,2);
#else
     ClassDef(Outer,3);
#endif
  };
}

//typedef HepExp::Inner Alias;

//#ifdef __MAKECINT__
//  #pragma read sourceClass="Alias" targetClass="HepExp::Inner";
//#endif

#include "TFile.h"

void writeFile(const char *filename = "nestedColl.root") 
{
   TFile *f = TFile::Open(filename,"RECREATE");
   HepExp::Outer obj(1);
   f->WriteObject(&obj,"obj");
   delete f;
}

void readFile(const char *filename = "nestedColl.root") 
{
   TFile *f = TFile::Open(filename,"READ");
   HepExp::Outer *obj;
   f->GetObject("obj",obj);
   if (obj) obj->Print();
   delete f;
}

int execNestedColl() {
   readFile();
   return 0;
}

