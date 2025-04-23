#include "TTree.h"

int runReadFile() {
   TTree *t = new TTree("t","tree created from readfile"); 
   t->ReadFile("data.txt","var1:var2"); 
   t->Scan("*");
   delete t;
   t = new TTree("arraytree","tree with array created from readfile");
   t->ReadFile("arraydata.txt");
   t->Scan();
   delete t;
   t = new TTree("arraytree","tree with array created from readfile");
   t->ReadFile("arraydata2.txt");
   t->Scan();
   delete t;
   t = new TTree("data","tree from csv");
   t->ReadFile("data.csv");
   t->Scan();
   delete t;
   t = new TTree("spectrum","tree from csv");
   t->ReadFile("spectrum.csv");
   t->Scan();
   delete t;
   return 0;
}
