#include <vector>
#include "TFile.h"

using namespace std;

class Reconstructor {};
class normal {};

class Holder : public TObject {
public:
   vector<const normal*> v1;
   vector<Reconstructor> v2;
   //vector<normal *const> v3;
   //vector<normal const*> v3;

   Holder() {};
   ~Holder() {};   

   ClassDef(Holder,1);
};

void constTest() {
   TFile *f = new TFile("test.root","RECREATE");
   Holder *h= new Holder;
   h->Write();
   f->Write();
   delete f;  
};
