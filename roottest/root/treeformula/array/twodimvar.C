#include "twodimvar.h"

#if !defined(__CINT__)
ClassImp(A);
#endif


A::A() {
  a=0;
  for(int k=0;k<3;k++) {
    aa[k]=0;
  }
  aaa = 0;
}


A::~A() {
  delete [] a;
  for(int k=0;k<3;k++) {
    delete [] aa[3-1-k];
  }
}

void A::Fill(int in) {
  n=in;
  if(a) delete [] a;
  for(int k=0;k<3;k++) {
    if(aa[3-1-k]) delete [] aa[3-1-k];
  }

  
  a = new int [n];
  for(int k=0;k<3;k++) {
    aa[k] = new int[n];
  }

  for(int i=0;i<n;i++) {
    a[i]=i+1;
  }
  for(int k=0;k<3;k++) {
    for(int i=0;i<n;i++) {
      aa[k][i] = (k+1)+(i+1)*10;
    }
  }
}

void A::Dump2() const {
  std::cout << n << std::endl;
  for(int i=0;i<n;i++) {
    std::cout << std::setw(5) << a[i];
  }
  std::cout << std::endl;
  
  for(int k=0;k<3;k++) {
    for(int i=0;i<n;i++) {
      std::cout << std::setw(5) << aa[k][i];
    }
    std::cout << std::endl;
  }
  std::cout << "--------------" << std::endl;
}


#include "TFile.h"
#include "TTree.h"

void twrite(const char* output,int splitlevel) 
{
  std::cout << "Output  file : " << output << std::endl;

  TFile* tfo = new TFile(output,"RECREATE");
  TTree* tree = new TTree("tree","test");
  
  A*   a = new A();

  tree->Branch("A","A",&a,32000,splitlevel);
  //  tree->Print();

  for (Int_t jentry=0; jentry<10;jentry++) {
    a->Fill(jentry+1);
    
    tree->Fill();
  }
  tree->Write();
  tfo->Close();
}

void tread(const char* input) {

  std::cout << "Input  file : " << input << std::endl;

  A*   a = new A();
  
  TFile* tfi = new TFile(input,"READ");
  TTree* tree = (TTree*)tfi->Get("tree");
  
  tree->SetBranchAddress("A",&a);

  Int_t nevent = Int_t(tree->GetEntries()); 

  for (Int_t jentry=0; jentry<nevent;jentry++) {

   Int_t ientry = tree->LoadTree(jentry);
   if (ientry < 0) break;
   tree->GetEntry(jentry);
   //a->Dump();
   a->Dump2();
  }
  tfi->Close();

  delete a;
  
}

void tscan(const char *input) {

   TFile* tf = new TFile(input);
   TTree* tree = (TTree*)tf->Get("tree");
   tree->Draw("aa[][]");
   tree->SetScanField(0);
   tree->Scan("aa");

}

void ss() {
   A a;
   fprintf(stdout,"A::aa=%ld A::aaa=%ld\n",
           (long)sizeof(a.aa),(long)sizeof(a.aaa));
}
