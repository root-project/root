#include <iostream>
#include <TTree.h>
#include <TFile.h>


int main(int argc, char* argv[]){
  TTree* ttree = new TTree("test","test");

  if(argc>1){
    ttree->ReadFile(argv[1],"",':');
  } else {
    ttree->ReadStream(std::cin,"",':');
  }
  
  ttree->Print();
  return 0;
}
