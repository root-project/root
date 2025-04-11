#include <vector>
#include "TTree.h"

class MyClass {
public:
   static Int_t fCounter;
   static Int_t fTotal;
   Int_t fSerial;
   MyClass(TRootIOCtor *) : fSerial(0) {
      ++fCounter;
      fprintf(stdout,"creating for I/O counter:%d \n",fCounter);
   }
   MyClass() : fSerial(++fTotal) {
      ++fCounter;
      fprintf(stdout,"creating serial:%d counter:%d \n",fSerial,fCounter); 
   };
   ~MyClass() { 
      fprintf(stdout,"delete serial:%d counter:%d \n",fSerial,fCounter); --fCounter; 
   }
};
Int_t MyClass::fCounter = 0;
Int_t MyClass::fTotal = 0;

#ifdef __MAKECINT__
#pragma link C++ class vector<MyClass*>;
#pragma link C++ class vector<vector<MyClass*> > ;
#pragma link C++ class vector<vector<MyClass*> *> ;
#endif

TTree *write()
{
   fprintf(stdout,"writing tree\n");
   TTree *tree = new TTree("T","T");

   vector<MyClass*> vec;
   vec.push_back(new MyClass());
   vec.push_back(new MyClass());

   vector<vector<MyClass*> > vecvec;
   vecvec.push_back(vector<MyClass*>());
   vecvec[0].push_back(new MyClass());
   vecvec[0].push_back(new MyClass());

   vector<vector<MyClass*> *> vecptrvec;
   vecptrvec.push_back(new vector<MyClass*>());
   vecptrvec[0]->push_back(new MyClass());
   vecptrvec[0]->push_back(new MyClass());

   tree->Branch("vec",&vec);
   tree->Branch("vecvec",&vecvec);
   tree->Branch("vecptrvec",&vecptrvec);
   tree->Fill();
   
   // Decrease size of vector.
   delete vec[0];
   delete vec[1];
   vec.clear();
   vec.push_back(new MyClass());

   delete vecvec[0][0];
   delete vecvec[0][1];
   vecvec[0].clear();
   vecvec[0].push_back(new MyClass());
   
   delete (*vecptrvec[0])[0];
   delete (*vecptrvec[0])[1];
   vecptrvec[0]->clear();
   vecptrvec[0]->push_back(new MyClass());
   
   tree->Fill();
   tree->ResetBranchAddresses();

   delete vec[0];
   delete vecvec[0][0];
   delete (*vecptrvec[0])[0];
   delete vecptrvec[0];

   return tree;
}

void read(TTree *tree) {
   fprintf(stdout,"reading tree\n");
   tree->GetEntry(0);
   tree->GetEntry(1);
   tree->GetEntry(0);
   tree->GetEntry(1);
}

Int_t execStlPtrLeak() {
   TTree *tree = write();
   read(tree);
   fprintf(stdout,"deleting tree\n");
   delete tree;
   fprintf(stdout,"Final counter %d\n",MyClass::fCounter);
   Bool_t good = (MyClass::fCounter == 0);
   fflush(stdout);
   if (good) {
      return 0;
   } else {
      return 1;
   }
}
