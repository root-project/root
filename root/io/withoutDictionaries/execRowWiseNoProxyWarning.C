// Test if writing an stl collection w/o proxy on a file issues an error.

#include "RowWiseNoProxyWarning.h"

void execRowWiseNoProxyWarning(){

   TFile ofile("execRowWiseNoProxyWarning.root","RECREATE");

   // It could be shorter with templates, but this is a test after all..

   // For these we have the dictionaries
   std::vector<CustomClass2> cVec2 = {1,2,3,4,5};
   ofile.WriteObject(&cVec2,"cVec2");

   std::list<CustomClass2> cList2 = {1,2,3,4,5};
   ofile.WriteObject(&cList2,"cList2");

   std::list<CustomClass2> cSet2 = {1,2,3,4,5};
   ofile.WriteObject(&cSet2,"cSet2");

   std::map<int, CustomClass2> cMap2 = {{1,1},{2,2},{3,3}};
   ofile.WriteObject(&cMap2,"cMap2");

   // For these we do not!

   std::vector<CustomClass> cVec = {1,2,3,4,5};
   ofile.WriteObject(&cVec,"cVec");

   std::list<CustomClass> cList = {1,2,3,4,5};
   ofile.WriteObject(&cList,"cList");

   std::list<CustomClass> cSet = {1,2,3,4,5};
   ofile.WriteObject(&cSet,"cSet");

   std::map<int, CustomClass> cMap = {{1,1},{2,2},{3,3}};
   ofile.WriteObject(&cMap,"cMap");

}
