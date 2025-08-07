///////////////////////////
//
// in new root session:
//     > .x macroFriends.C
//
///////////////////////////

#include "TBranch.h"
#include "TFile.h"
#include "TFriendElement.h"
#include "TLeaf.h"
#include "TList.h"
#include "TRandom.h"
#include "TTree.h"
#include "TString.h"

#include <iostream>


class MyClass: public TNamed {

   protected:
      TFile   *fFile;
      TTree   *fTree;
      TTree   *fTreeY;
      TTree   *fTreeZ;

   public:
      MyClass() {}
      MyClass(const char *name, const char *title = "test");
      virtual ~MyClass();

      void CreateTrees(const char *treename, const char *filename);
      void AddTree(const char *treename, const char *filename);
      void CopyTrees1(const char *treename, const char *filename);
      void CopyTrees2(const char *treename, const char *filename);
      void GetTree();
      void GetTreeZ();

#if !defined (__CINT__) || defined (__MAKECINT__)
      ClassDefOverride(MyClass,1) //MyClass
#endif
};

class MyData {
//class MyData: public TObject {

   protected:
      Int_t    fID;
      Double_t fX;

   public:
      MyData() {}
      virtual ~MyData() {}

      void SetID(Int_t id)   {fID = id;}
      void SetX(Double_t x)  {fX  = x;}

      Int_t    GetID() const {return fID;}
      Double_t GetX()  const {return fX;}

#if !defined (__CINT__) || defined (__MAKECINT__)
      ClassDef(MyData,1) //MyData
#endif
};


#if !defined (__CINT__) || defined (__MAKECINT__)
ClassImp(MyClass);
ClassImp(MyData);
#endif

//______________________________________________________________________________
MyClass::MyClass(const char *name, const char *title)
        :TNamed(name, title)
{
   cout << "------MyClass::MyClass------" << endl;

   fFile  = 0;
   fTree  = 0;
   fTreeY = 0;
   fTreeZ = 0;
}//Constructor

//______________________________________________________________________________
MyClass::~MyClass()
{
   cout << "------MyClass::~MyClass------" << endl;

   SafeDelete(fTreeZ);
   SafeDelete(fTreeY);
   SafeDelete(fTree);
   SafeDelete(fFile);
}//Destructor

//______________________________________________________________________________
void MyClass::CreateTrees(const char *treename, const char *filename)
{
   cout << "------MyClass::CreateTrees------" << endl;

   TFile *file = new TFile(filename,"RECREATE");

// Number  of tree entries
   Int_t nentries = 100;

   TTree *tree = 0;
   TString str = "";

   gRandom->SetSeed(111);
   for (Int_t i=0; i<4; i++) {
      str = treename; str += i;
      tree = new TTree(str, "trees");

      Int_t split = 99;
      MyData *data = 0;
      data = new MyData();
      tree->Branch("DataBranch", "MyData", &data, 64000, split);

      for (Int_t j=0; j<nentries; j++) {
         data->SetID(j);
         data->SetX(gRandom->Rndm(1));
         tree->Fill();
      }//for_j

      tree->Write();
   }//for_i

   delete file;
}//CreateTrees

//______________________________________________________________________________
void MyClass::AddTree(const char *treename, const char *filename)
{
   cout << "------MyClass::AddTree------" << endl;

   if (!fFile) fFile = new TFile(filename,"READ");

   if (!fTree) fTree = (TTree*)fFile->Get(treename);
   else        fTree->AddFriend(treename, filename);

}//AddTree

//______________________________________________________________________________
void MyClass::CopyTrees1(const char *treename, const char *filename)
{
   // creates new trees and copies selected entries
   cout << "------MyClass::CopyTrees1------" << endl;

   TList *friends  = fTree->GetListOfFriends();
   Int_t  nentries = (Int_t)(fTree->GetEntries());
   Int_t  nfriends = friends->GetSize();
   Int_t  ntrees   = nfriends + 1;
cout << "nfriends(fTree) = " << nfriends << endl;
cout << "nentries(fTree) = " << nentries << endl;

   TFriendElement *fe = 0;
   TTree   *treej[20];
   MyData  *dataj[20];
   for (Int_t j=0; j<ntrees; j++) {
      dataj[j] = 0;
   }//for_j

// Create tree/branch/leaf arrays
   fFile->cd();
   fe = (TFriendElement*)friends->At(0);
   treej[0] = fe->GetParentTree();
cout << "treej[0] = " << treej[0]->GetName() << endl;
   treej[0]->SetBranchAddress("DataBranch",&dataj[0]);
   for (Int_t j=0; j<nfriends; j++) {
      fe = (TFriendElement*)friends->At(j);
      treej[j+1] = fe->GetTree();
cout << "treej[" << j+1 << "] = " << treej[j+1]->GetName() << endl;
      treej[j+1]->SetBranchAddress("DataBranch",&dataj[j+1]);
   }//for_j

   TFile *file = new TFile(filename,"RECREATE");
cout << "file = " << file->GetName() << endl;


   TTree *newtree = 0;
   TTree *tmptree = 0;

   for (Int_t j=0; j<ntrees; j++) {
//      newtree = treej[j]->CloneTree(0);
      TString name  = treej[j]->GetName();
      TString alias = name + "=" + TString(treej[j]->GetName());
cout << "alias = " << alias << endl;
      tmptree = new TTree(treej[j]->GetName(), treej[j]->GetTitle());

      Int_t split = 99;
      MyData *data = 0;
      data = new MyData();
      tmptree->Branch("DataBranch", "MyData", &data, 64000, split);

      for (Int_t i=0; i<nentries; i++) {
         // Test: filter mask!!!!
         if (i%2) continue;

         treej[j]->GetEntry(i);
         data->SetID(dataj[j]->GetID());
         data->SetX(dataj[j]->GetX());
if (i<4) cout << "i= " << i <<  " j= " << j << "  data= " << data->GetX() << endl;
         tmptree->Fill();
      }//for_i

      tmptree->Write();

      if (j == 0) {
         newtree = tmptree;
      } else {
         newtree->AddFriend(tmptree, alias.Data());

//         delete tmptree;
//         tmptree->Delete(""); tmptree = 0;
      }//if

      delete data;
   }//for_j

   fTree = newtree;
//   fTreeZ = newtree;
}//CopyTrees1

//______________________________________________________________________________
void MyClass::CopyTrees2(const char *treename, const char *filename)
{
   cout << "------MyClass::CopyTrees2------" << endl;

   TList *friends  = fTree->GetListOfFriends();
   Int_t  nentries = (Int_t)(fTree->GetEntries());
   Int_t  nfriends = friends->GetSize();
   Int_t  ntrees   = nfriends + 1;
cout << "nfriends(fTree) = " << nfriends << endl;
cout << "nentries(fTree) = " << nentries << endl;

   TFriendElement *fe = 0;
   TTree   *treej[20];
   MyData  *dataj[20];
   for (Int_t j=0; j<ntrees; j++) {
      dataj[j] = 0;
   }//for_j

// Create tree/branch/leaf arrays
   fFile->cd();
   fe = (TFriendElement*)friends->At(0);
   treej[0] = fe->GetParentTree();
cout << "treej[0] = " << treej[0]->GetName() << endl;
   treej[0]->SetBranchAddress("DataBranch",&dataj[0]);
   for (Int_t j=0; j<nfriends; j++) {
      fe = (TFriendElement*)friends->At(j);
      treej[j+1] = fe->GetTree();
cout << "treej[" << j+1 << "] = " << treej[j+1]->GetName() << endl;
      treej[j+1]->SetBranchAddress("DataBranch",&dataj[j+1]);
   }//for_j

   TFile *file = new TFile(filename,"RECREATE");
cout << "file = " << file->GetName() << endl;

   TTree *tree = 0;

   for (Int_t j=0; j<ntrees; j++) {
      tree = treej[j]->CloneTree(0);
cout << "tree[" << j << "]= " << tree->GetName() << endl;

      for (Int_t i=0; i<nentries; i++) {
         // Test: select every second entry
         if (i%2) continue;

         treej[j]->GetEntry(i);
if (i<4) cout << "j= " << j << "  x[" << i << "]= " << dataj[j]->GetX() << endl;
         tree->Fill();
      }//for_j

      tree->Write();

      if (j == 0) {
         fTreeZ = tree;
      } else {
//not allowed!         fTreeZ->AddFriend(tree, str.Data());

         delete tree;
      }//if
   }//for_i
}//CopyTrees2

//______________________________________________________________________________
void MyClass::GetTree()
{
   cout << "------MyClass::GetTree------" << endl;

   if (!fTree) {cout << "fTree does not exist" << endl; return;}

   TList *friends  = fTree->GetListOfFriends();
   Int_t  nentries = (Int_t)(fTree->GetEntries());
   Int_t  nfriends = friends->GetSize();

   cout << "fTreeName  = " << fTree->GetName() << endl;
   cout << "fTreeTitle = " << fTree->GetTitle() << endl;
   cout << "nentries(tree) = " << nentries << endl;
   cout << "nfriends(tree) = " << nfriends << endl;
}//GetTree

//______________________________________________________________________________
void MyClass::GetTreeZ()
{
   cout << "------MyClass::GetTreeZ------" << endl;

   if (!fTreeZ) {cout << "fTreeZ does not exist" << endl; return;}

   TList *friends  = fTreeZ->GetListOfFriends();
   Int_t  nentries = (Int_t)(fTreeZ->GetEntries());
   Int_t  nfriends = friends->GetSize();

   cout << "nentries(treeZ) = " << nentries << endl;
   cout << "nfriends(treeZ) = " << nfriends << endl;
}//GetTreeZ

//______________________________________________________________________________
//______________________________________________________________________________
void macroFriends2()
{
   MyClass *myclass = new MyClass("MyClass");

   myclass->CreateTrees("TreeX","TreeX.root");
   myclass->CreateTrees("TreeY","TreeY.root");

   myclass->AddTree("TreeX0","TreeX.root");
   myclass->AddTree("TreeX1","TreeX.root");
   myclass->AddTree("TreeX2","TreeX.root");
   myclass->AddTree("TreeX3","TreeX.root");
   myclass->AddTree("TreeY0","TreeY.root");
   myclass->AddTree("TreeY1","TreeY.root");
   myclass->AddTree("TreeY2","TreeY.root");
   myclass->AddTree("TreeY3","TreeY.root");

   myclass->CopyTrees1("TreeZ","TreeZ1.root");
   myclass->GetTree();
//   myclass->GetTreeZ();

   myclass->CopyTrees2("TreeZ","TreeZ2.root");
//   myclass->GetTree();
   myclass->GetTreeZ();

   delete myclass;
}//macroFriends
