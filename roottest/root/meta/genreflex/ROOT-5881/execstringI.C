#include "TFile.h"
#include "TTree.h"
#include <string>
#include <iostream>

int readTree(){

  TFile myFile("stringTree.root");
  TTree* myTree = (TTree*) myFile.GetObjectChecked("stringTree","TTree");
  if (!myTree) return 1;
  std::string* myString = new std::string();
  myTree->SetBranchAddress("myString", &myString);

  for (int i=0;i<myTree->GetEntries();++i){
    myTree->GetEntry(i);
    std::cout << "Entry " << i << " " << *myString << std::endl;
    }
  myFile.Close();
  return 0;
  
}

int execstringI(){
 return readTree();
 
}

