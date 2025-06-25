#include <stdlib.h>
#include "TApplication.h"
#include "TTree.h"
#include "TChain.h"
#include "TFile.h"
#include "TEntryListArray.h"
#include "TRandom.h"
#include "TDirectory.h"
#include "TCut.h"
#include "TObjArray.h"
#include "TChainElement.h"

#include <vector>
#include <iostream>

using namespace std;

/** Return a vector with the values given by Tree::Draw **/
std::vector<Double_t> getValues(Double_t* x, Int_t N)
{
  std::vector<Double_t> v(N);
  for (Int_t i = 0; i < N; ++i)
    v[i] = x[i];
  return v;
}

/** Create trees for testing TEntryListArray (Int_t nentries, Int_t nfiles, Int_t maxSubEntries = 100, Int_t Ntrees = 2) **/
void MakeTrees(Int_t nentries, Int_t nfiles, Int_t Ntrees = 2, Int_t maxSubEntries = 100)
{
   TFile *f1;
   std::vector< TTree* > trees(Ntrees);

   std::vector<double> x, y, z;
   Double_t range = nentries/100.;

   for (Int_t ifile=0; ifile<nfiles; ifile++) {
      f1 = new TFile(Form("testEntryListTrees_%d.root", ifile), "UPDATE");
      // Create trees
      for (Int_t itree=0; itree < Ntrees; ++itree) {
        trees[itree] = new TTree(Form("tree%d", itree+1), Form("tree%d", itree+1));
        trees[itree]->Branch("x", &x);
        trees[itree]->Branch("y", &y);
        trees[itree]->Branch("z", &z);
      }
      // Fill trees
      for (Int_t i=0; i<nentries; i++){
        for (Int_t itree=0; itree < Ntrees; ++itree) {
          x.clear();
          y.clear();
          z.clear();
          Int_t NsubEntries = gRandom->Integer(maxSubEntries+1);
          for (Int_t j=0; j < NsubEntries; ++j) {
            x.push_back(gRandom->Uniform(-range, range));
            y.push_back(gRandom->Uniform(-range, range));
            z.push_back(gRandom->Uniform(-range, range));
          }
          trees[itree]->Fill();
        }
      }
      // Write trees
      for (Int_t itree=0; itree < Ntrees; ++itree) trees[itree]->Write();
      f1->Close();
   }
}

/** Check the entries and subentries stored in TEntryListArray using TChain / TTree::Draw
 @param cut the cut to be used for the test
 @param elist a list produced using the cut (optional)
 **/
Bool_t TestSelection(TTree *tree, TCut cut, TEntryListArray *elist = 0)
{
  tree->SetEntryList(0);
  if (!elist)
  {
    tree->Draw(">> elist", cut, "entrylistarray");
    elist = (TEntryListArray*) gDirectory->Get("elist");
  }

  // Check the number of entries
  if (!elist || elist->GetN() != tree->GetEntries(cut)) return false;

  // Check the total number of subentries
  Int_t NsubEntries = tree->Draw("x", cut, "goff");
  std::vector< Double_t > x1 = getValues(tree->GetV1(), NsubEntries);

  tree->SetEntryList(elist);
  if (tree->Draw("x", "", "goff") != NsubEntries) return false;
  std::vector< Double_t > x2 = getValues(tree->GetV1(), NsubEntries);
  tree->SetEntryList(0);

  if (x1 != x2) return false;

  return true;
}

/** Test the selection using TEntryListArray::Contains **/ // BUG: Does not work with chains
// Bool_t TestSelectionWithContains(TTree *tree, TCut cut, TEntryListArray *elist = 0)
// {
//   std::vector<int> v;
//   tree->SetEntryList(0);
//   if (!elist)
//   {
//     TObject *obj = gDirectory->Get("elist");
//     if (obj)
//       obj->Delete();
//     tree->Draw(">> elist", cut, "entrylistarray");
//     elist = (TEntryListArray*) gDirectory->Get("elist");
//   }
//
//   Int_t NsubEntries = tree->Draw("Entry$:Iteration$", cut, "goff");
//   for (Int_t i =0; i < NsubEntries; ++i) {
//     if (!elist->Contains((Long64_t) tree->GetV1()[i], 0, (Long64_t) tree->GetV2()[i]))
//       return false;
//   }
//
//   return true;
// }




/** Return true if the given TEntryListArray objects have the same entries **/
// Bool_t CompareLists(TEntryListArray *e1, TEntryListArray *e2)
// {
//   if (!e1 && !e2) return true;
//   else if (!e1 || !e2) return false;
//
//   if (e1->GetN() != e2->GetN())
//     return false;
//
//   // The lists are not splitted, compare the entries and subentries
//   if (!e1->GetLists() && !e2->GetLists())
//   {
//     for (Int_t i = 0; i < e1->GetN(); ++i)
//     {
//       Int_t entry = e1->GetEntry(i);
//       if (e2->GetEntry(i) != entry)
//         return false;
//       // Compare subentries
//       if (!CompareLists(e1->GetSubListForEntry(entry), e2->GetSubListForEntry(entry)))
//         return false;
//     }
//   }
//
//  // Only e2 is splitted,
//   else if (!e1->GetLists())
//     return CompareLists(e2, e1); // Invert the order to avoid duplicating the code
//
//   // e1 is splitted (and maybe also e2)
//   else {
//     TEntryListArray* e = 0;
//     TIter next(e1->GetLists());
//     while ((e = (TEntryListArray*) next())) {
//       e2->SetTree(e->GetTreeName(), e->GetFileName());
//       if (!CompareLists(e, (TEntryListArray*) e2->GetCurrentList()))
//         return false;
//     }
//   }
//
//   return true;
// }


/** Test copy constructor of TEntryListArray **/
Bool_t TestCopy(TTree *tree, TCut cut)
{
  tree->Draw(">> e1", cut, "entrylistarray");
  TEntryListArray *e1 = (TEntryListArray*) gDirectory->Get("e1");

  TEntryListArray *e2 = new TEntryListArray(*e1);
  Bool_t result = TestSelection(tree, cut, e2);
  return result;
}


/** Test cloning TEntryListArrays **/
Bool_t TestClone(TTree *tree, TCut cut)
{
  tree->Draw(">> e1", cut, "entrylistarray");
  TEntryListArray *e1 = (TEntryListArray*) gDirectory->Get("e1");

  TEntryListArray *e2 = (TEntryListArray*) e1->Clone("e2");
  Bool_t result = TestSelection(tree, cut, e2);
  return result;
}


/** Test adding and subtracting TEntryListArrays in different orders **/
Bool_t TestAddAndSubtract(TTree *tree, TCut c1, TCut c2)
{
  tree->Draw(">> e1", c1, "entrylistarray");
  TEntryListArray *e1 = (TEntryListArray *) gDirectory->Get("e1");

  tree->Draw(">> e2", c2, "entrylistarray");
  TEntryListArray *e2 = (TEntryListArray *) gDirectory->Get("e2");
  if (!e1 || !e2)   {
    cout << "Could not retreive lists" << endl;
    return false;
  }

  TEntryListArray *eA = new TEntryListArray(*e1);
  TEntryListArray *eB = new TEntryListArray(*e2);

  eA->Add(e2);
  if (!TestSelection(tree, c1 || c2, eA ))
    return false;

  eB->Add(e1);
  if (!TestSelection(tree, c1 || c2, eB ))
    return false;

  return true;
}


/** Test adding and subtracting TEntryListArrays **/
Bool_t stressAddAndSubtract(TTree *tree)
{
  Bool_t result = true;
  if (!TestAddAndSubtract(tree, "x>1", "x < -1"))
  {
    result = false;
    cout << "Add and Subtract without intersection failed" << endl;
  }

  if (!TestAddAndSubtract(tree, "x> 1", "x > 0.5 && x < 2"))
  {
    result = false;
    cout << "Add and Subtract with intersection failed" << endl;
  }

  tree->Draw(">> elist", "x>1", "entrylistarray");
  tree->Draw(">>+ elist", "x<-1", "entrylistarray");
  if (!TestSelection(tree, "x> 1 || x<-1", (TEntryListArray*) gDirectory->Get("elist")))
  {
    result = false;
    cout << "Add and Subtract with with >>+ operator failed" << endl;
  }


  TChain *chain = dynamic_cast<TChain*>(tree);
  if (chain)
  {
    TCut cut = "x>1";
    TEntryListArray *elist = new TEntryListArray();
    TObjArray *fileElements=chain->GetListOfFiles();
    TIter next(fileElements);
    TChainElement *chEl=0;
    while (( chEl=(TChainElement*)next() )) {
      TFile f(chEl->GetTitle());
      TTree *tree1 = (TTree*) f.Get(chain->GetName());
      tree1->Draw(">> elTest", cut, "entrylistarray");
      elist->Add((TEntryListArray*) gDirectory->Get("elTest") );
    }

    if (!TestSelection(chain, cut, elist))
    {
        cout << "Add with different trees failed" << endl;
        result = false;
    }
  }

  return result;
}


int execTEntryListArray(Int_t nentries=1000, Int_t nfiles=3) {
  cout << "***************************************" << endl;
  cout << "*****   Testing TEntryListArray   *****" << endl;
  cout << "***************************************" << endl;

  MakeTrees(nentries, nfiles);
  cout << "Tree making done" << endl;

  TChain *chain = new TChain("tree1");
  chain->Add("testEntryListTrees_*.root");

  Bool_t result = true;

  if (!(result = TestSelection(chain, "x>0")))
    cout << "SelectionWithChain failed" << endl;
  if (!(result = TestCopy(chain, "x>1")))
    cout << "TestCopy failed" << endl;
  if (!(result = stressAddAndSubtract(chain)))
    cout << "AddAndSubtract failed" << endl;
  if (!(result = TestClone(chain, "x>0")))
    cout << "Clone failed" << endl;

  cout << "Result of tests with TEntryListArray: " << result << endl;
  cout << "***************************************" << endl;

  return (result != true);
}

//_____________________________batch only_____________________
#ifndef __CLING__

int main(int argc, char *argv[])
{
   TApplication theApp("App", &argc, argv);
   Int_t nentries = 1000;
   Int_t nfiles = 3;
   if (argc > 1) nentries = atoi(argv[1]);
   if (argc > 2) nfiles = atoi(argv[2]);
   execTEntryListArray(nentries, nfiles);
   return 0;
}

#endif
