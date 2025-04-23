#include "TFile.h"
#include "TTree.h"

#include <string>

void writeTree(){

  const std::string myStringTempl("myString_");
  std::string myString;

  TFile ofile("stringTree.root","RECREATE");
  TTree theTree("stringTree","stringTree");
  theTree.Branch("myString",&myString);

  for (int i=0;i<10;i++){
    myString=myStringTempl;
    char c[3];
    sprintf(c, "%d", i);
    myString+=c;
    theTree.Fill();
    }
  theTree.Write();
  ofile.Close();
}

int main(){
 writeTree();
}
