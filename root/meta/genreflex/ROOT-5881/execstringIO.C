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

int writeTree(){

  const std::string myStringTempl("myString_");
  std::string myString;

  TFile ofile("stringTree.root","RECREATE");
  TTree theTree("stringTree","stringTree");
  theTree.Branch("myString",&myString);

  for (int i=0;i<100;i++){
    myString=myStringTempl;
    char c[3];
    sprintf(c, "%d", i);
    myString+=c;
    theTree.Fill();
    }
  theTree.Write();
  ofile.Close();
  return 0;
}

int execstringIO(){
 int writeRet = writeTree();
 int readRet = readTree();
 return writeRet + readRet;
 
}
