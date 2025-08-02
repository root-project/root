#include "TFile.h"
#include "TTree.h"
#include <iostream>
#include <vector>
#include <string>

void print(std::vector<std::string> &what)
{
   cout << "Printing vector\n";
   for(unsigned int i=0;i<what.size(); ++i) {
      std::cout << i << ":";
      std::cout << what[i];
      std::cout << '\t';
   }
   std::cout << std::endl;
}

void runvstr() {
   std::vector<std::string> RProcess;
   RProcess.reserve(1000);

   std::string tmp("two");
   std::cout << tmp << std::endl;
   RProcess.push_back(tmp);
   RProcess.push_back("123");

   print(RProcess);

   TTree *t1 = new TTree("t1","t1");
   t1 -> Branch("RProcess", &RProcess);
   t1->Fill();

   RProcess.clear();
   print(RProcess);

   t1->GetEntry(0);

   print(RProcess);

   t1->Scan("RProcess.c_str()");
   t1->ResetBranchAddresses();
}

