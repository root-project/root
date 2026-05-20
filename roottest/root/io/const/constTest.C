#include <vector>
#include "TFile.h"

using namespace std;

class Reconstructor {};
class Normal {};

class Holder : public TObject {
public:
   vector<const Normal*> v1;
   vector<Reconstructor> v2;
   //vector<Normal *const> v3;
   //vector<Normal const*> v3;

   Holder() {};
   ~Holder() {};   

   ClassDefOverride(Holder,1);
};

void constTest() {
   TFile *f = new TFile("test.root","RECREATE");
   Holder *h= new Holder;
   h->Write();
   f->Write();
   delete f;  
};
