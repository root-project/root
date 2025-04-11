class Simple {
public:
   Simple(int i = -1) : value(i) {}
   int value;
};

#include "TTree.h"

void runnodot() {
   Simple s1(22);
   Simple s2(33);
   Simple s3(44);

   TTree *tree = new TTree("tree","title");
   tree->Branch("s1",&s1);
   tree->Branch("s2",&s2);
   tree->Branch("s3",&s3);
   tree->Branch("sdot1.",&s1);
   tree->Branch("sdot2.",&s2);
   tree->Branch("sdot3.",&s3);
   tree->Fill();
   tree->Scan("*");
   tree->Scan("s1.value:s2.value:s3.value");
   tree->Scan("sdot1.value:sdot2.value:sdot3.value");
}
