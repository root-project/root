#include "IOWithoutDictionaries.h"
#include "TFile.h"
#include "TTree.h"
#include <iostream>
using namespace std;

// We have a matrix:
// o classWithDictionary1, classWithDictionary1 have a dictionary
// o classWithoutDictionary1, classWithoutDictionary1 DO NOT have a dictionary
// o vector<classWithDictionary1 and vector<classWithoutDictionary1 have a dictionary

class ObjectsForIO {
public:
   ObjectsForIO():
      cwd1Instance(-1),
      cwd1Instances({{classWithDictionary1(-1), classWithDictionary1(-2), classWithDictionary1(-3)}}),
   cwod1Instance(-1),
   cwod1Instances({{classWithoutDictionary1(-1), classWithoutDictionary1(-2), classWithoutDictionary1(-3)}}) {};

//------------------------------------------------------------------------------

   ObjectsForIO(TFile &ifile) {
      auto cwd1InstanceLocal = &cwd1Instance;
      ifile.GetObject("cwd1Instance", cwd1InstanceLocal);
      cwd1Instance = *cwd1InstanceLocal;

      auto cwd1InstancesLocal = &cwd1Instances;
      ifile.GetObject("cwd1Instances", cwd1InstancesLocal);
      if (cwd1InstancesLocal){
         cwd1Instances = *cwd1InstancesLocal;
      } else {
         cout << "Error: no data could be read for cwd1Instances\n";
      }


      auto cwod1InstanceLocal = &cwod1Instance;
      ifile.GetObject("cwod1Instance", cwod1InstanceLocal);
      cwod1Instance = *cwod1InstanceLocal;

      auto cwod1InstancesLocal = &cwod1Instances;
      ifile.GetObject("cwod1Instances", cwod1InstancesLocal);

      if (cwod1InstancesLocal){
         cwod1Instances = *cwod1InstancesLocal;
      } else {
         cout << "Error: no data could be read for cwod1Instances\n";
      }

   };

//------------------------------------------------------------------------------

   ObjectsForIO(const classWithDictionary1 &otherCwd1Instance,
                const vector<classWithDictionary1> &otherCwd1Instances,
                const classWithoutDictionary1 &otherCwod1Instance,
                const vector<classWithoutDictionary1> &otherCwod1Instances):
      cwd1Instance(otherCwd1Instance),
      cwd1Instances(otherCwd1Instances),
      cwod1Instance(otherCwod1Instance),
      cwod1Instances(otherCwod1Instances) {}

//------------------------------------------------------------------------------

int CompareRowWise(const ObjectsForIO &other, bool verbose=true) const {

      int ret = 0;

      if (cwd1Instance != other.cwd1Instance) {
         ret+=1;
         if (verbose) cout << "ERROR: classWithDictionary1 instances are different:"
              << cwd1Instance.GetI() << " " << other.cwd1Instance.GetI() << endl;
      }
      if (cwod1Instance != other.cwod1Instance) {
         ret+=1;
         if (verbose) cout << "ERROR: classWithoutDictionary1 instances are different:"
              << cwod1Instance.GetI() << " " << other.cwod1Instance.GetI() << endl;
      }

      if (cwd1Instances.size() != other.cwd1Instances.size()) {
         ret+=1;
         if (verbose) cout << "ERROR: vector<classWithDictionary1> instances have different sizes: "
              << cwd1Instances.size() << " " <<  other.cwd1Instances.size() << endl;
      } else {
         for (int i = 0; i < cwd1Instances.size(); i++) {
            if (cwd1Instances[i] != other.cwd1Instances[i]) {
               ret+=1;
               if (verbose) cout << "ERROR: vector<classWithDictionary1> instance element "
                    << i << " differs: " << cwd1Instances[i].GetI() << " " << other.cwd1Instances[i].GetI() << endl;
            }
         }
      }

      if (cwod1Instances.size() != other.cwod1Instances.size()) {
         ret+=1;
         if (verbose) cout << "ERROR: vector<classWithoutDictionary1> instances have different sizes: "
              << cwod1Instances.size() << " " <<  other.cwod1Instances.size() << endl;
      } else {
         for (int i = 0; i < cwod1Instances.size(); i++) {
            if (cwod1Instances[i] != other.cwod1Instances[i]) {
               ret+=1;
               if (verbose) cout << "ERROR: vector<classWithoutDictionary1> instance element "
                    << i << " differs: " << cwod1Instances[i].GetI() << " " << other.cwod1Instances[i].GetI() << endl;
            }
         }
      }
      return ret;
   }

//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// Data
   classWithDictionary1 cwd1Instance;
   vector<classWithDictionary1> cwd1Instances;
   classWithoutDictionary1 cwod1Instance;
   vector<classWithoutDictionary1> cwod1Instances;
};

void readWithoutDictionaries()
{
   auto *ifilePtr = TFile::Open("ofileWithoutDictionaries.root");
   auto &ifile = *ifilePtr;

   // Row wise

   ObjectsForIO refObjs;
   ObjectsForIO readObjs(ifile);
   refObjs.CompareRowWise(readObjs,false);

   // Column wise
   TTree *tree;
   ifile.GetObject("tree", tree);
   if (!tree) {
      cout << "The tree could not be read\n";
      return;
   }

   ObjectsForIO objsFromTree;
   auto cwd1InstancePtr(&objsFromTree.cwd1Instance);
   auto cwd1InstancesPtr(&objsFromTree.cwd1Instances);
   auto cwod1InstancePtr(&objsFromTree.cwod1Instance);
   auto cwod1InstancesPtr(&objsFromTree.cwod1Instances);

   tree->SetBranchAddress("cwd1Instance", &cwd1InstancePtr);
   tree->SetBranchAddress("cwd1Instances", &cwd1InstancesPtr);
   tree->SetBranchAddress("cwod1Instance", &cwod1InstancePtr);
   tree->SetBranchAddress("cwod1Instances", &cwod1InstancesPtr);

   int ret = 0;
   for (int i = 0; i < 100; i++) {
      if (i % 10 == 0) std::cout << "Entry " << i << std::endl;
      tree->GetEntry(i);
      refObjs.cwd1Instance.SetI(i * i);
      refObjs.cwod1Instance.SetI(i * i);
      for (int j = 0; j < i % 10; ++j) {
         refObjs.cwd1Instances.emplace_back(j);
         refObjs.cwod1Instances.emplace_back(j);
      }

      ret += refObjs.CompareRowWise(objsFromTree,false);
      refObjs.cwd1Instances.clear();
      refObjs.cwod1Instances.clear();

   }

   if (ret!=0){
      cout << "Differences detected between the written and expected objects (this is expected)\n";
   }

}

void writeWithoutDictionaries()
{

   auto ofilePtr = TFile::Open("ofileWithoutDictionaries.root", "RECREATE");
   auto &ofile = *ofilePtr;

   cout << "Row-wise streaming\n";
   ObjectsForIO objs;

   ofile.WriteObjectAny(&objs.cwd1Instance, "classWithDictionary1", "cwd1Instance");
   ofile.WriteObjectAny(&objs.cwd1Instances, "vector<classWithDictionary1>", "cwd1Instances");
   ofile.WriteObjectAny(&objs.cwod1Instance, "classWithoutDictionary1", "cwod1Instance");
   ofile.WriteObjectAny(&objs.cwod1Instances, "vector<classWithoutDictionary1>", "cwod1Instances");

   cout << "Column-wise streaming\n";

   TTree tree("tree", "A Tree with classes with and without dictionaries");
   classWithDictionary1 cwd1Instance;
   vector<classWithDictionary1> cwd1InstancesTree;
   classWithoutDictionary1 cwod1Instance;
   vector<classWithoutDictionary1> cwod1InstancesTree;

   tree.Branch("cwd1Instance", &cwd1Instance);
   tree.Branch("cwd1Instances", &cwd1InstancesTree);
   tree.Branch("cwod1Instance", &cwod1Instance);
   tree.Branch("cwod1Instances", &cwod1InstancesTree);

   for (int i = 0; i < 100; ++i) {
      cwd1Instance.SetI(i * i);
      cwod1Instance.SetI(i * i);
      for (int j = 0; j < i % 10; ++j) {
         cwd1InstancesTree.emplace_back(j);
         cwod1InstancesTree.emplace_back(j);
      }

      tree.Fill();
      cwd1InstancesTree.clear();
      cwod1InstancesTree.clear();
   }

   tree.Write();
   ofile.Close();

   delete ofilePtr;

}

void execIOWithoutDictionaries()
{
   writeWithoutDictionaries();
   readWithoutDictionaries();
}
