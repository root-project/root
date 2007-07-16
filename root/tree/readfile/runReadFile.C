#include "TTree.h"

int runReadFile() {
   TTree *t = new TTree("t","tree created from readfile"); 
   t->ReadFile("data.txt","var1:var2"); 
   t->Scan("*");
   return 0;
}
