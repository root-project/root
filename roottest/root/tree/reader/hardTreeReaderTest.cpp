#include "TFile.h"
#include "TTreeReader.h"
#include "TChain.h"
#include "TInterpreter.h"
#include "TSystem.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include <vector>
#include "A.h"

#define TREE_ENTRIES 10
#define LIST_ENTRIES 10

#define NUM_CONSTANT 14
#define MULTIPLIER_B_OBJECT 1
#define MULTIPLIER_VECTOR_B 1
#define MULTIPLIER_VECTOR_FLOAT 1
#define MULTIPLIER_VECTOR_B_STAR 3
#define MULTIPLIER_VECTOR_STAR_B 2
#define MULTIPLIER_B_STAR 2
#define MULTIPLIER_B_ARRAY 4
#define MULTIPLIER_B_STAR_ARRAY 5
#define MULTIPLIER_B_CLONES_ARRAY 6
#define MYDOUBLEARRAY_SIZE 8
#define MYBOOLARRAYB_SIZE 12

void makeTree(const char* fileName, Int_t startI = 1){
    TFile *myFile = new TFile(fileName, "RECREATE");
    TTree *myTree = new TTree("HardTree", "This is hard to read");

    A myObject0 (NUM_CONSTANT);

    struct {
        Float_t myFloatX;
        Float_t myFloatY;
        Int_t myIntN;
        Bool_t myBoolArrayB [MYBOOLARRAYB_SIZE];
        Double_t myDoubleArrayA [MYDOUBLEARRAY_SIZE];
    } myLeaves;

    myLeaves.myFloatX = 0.0;
    myLeaves.myFloatY = 0.0;
    myLeaves.myIntN = MYDOUBLEARRAY_SIZE;

    myTree->Branch("A0.",   "A",    &myObject0,     32000, 0);
    myTree->Branch("A1.",   "A",    &myObject0,     32000, 1);
    myTree->Branch("A99.",  "A",    &myObject0, 32000, 99);
    myTree->Branch("A101.", "A",    &myObject0, 32000, 101);

    myTree->Branch("S0_num",            &myObject0.num, 32000, 0);
    myTree->Branch("S1_num",            &myObject0.num, 32000, 1);
    myTree->Branch("S99_num",           &myObject0.num, 32000, 99);
    myTree->Branch("S101_num",          &myObject0.num, 32000, 101);

    myTree->Branch("S0_vectorFloat",    &myObject0.vectorFloat, 32000, 0);

    myTree->Branch("S0_vectorB",        &myObject0.vectorB, 32000, 0);
    myTree->Branch("S1_vectorB",        &myObject0.vectorB, 32000, 1);
    myTree->Branch("S99_vectorB",       &myObject0.vectorB, 32000, 99);
    // myTree->Branch("S101_vectorB",       &myObject0.vectorB, 32000, 101); // Breaks Fill()

    myTree->Branch("S0_vectorBStar",    &myObject0.vectorBStar, 32000, 0);
    myTree->Branch("S1_vectorBStar",    &myObject0.vectorBStar, 32000, 1);
    myTree->Branch("S99_vectorBStar",   &myObject0.vectorBStar, 32000, 99);
    // myTree->Branch("S101_vectorBStar",   &myObject0.vectorBStar, 32000, 101); // Breaks Fill()

    myTree->Branch("S0_vectorStarB",    &myObject0.vectorStarB, 32000, 0);
    myTree->Branch("S1_vectorStarB",    &myObject0.vectorStarB, 32000, 1);
    myTree->Branch("S99_vectorStarB",   &myObject0.vectorStarB, 32000, 99);
    // myTree->Branch("S101_vectorStarB",   &myObject0.vectorStarB, 32000, 101); // Breaks Fill()

    myTree->Branch("S0_BStar",          &myObject0.BStar,   32000, 0);
    myTree->Branch("S1_BStar",          &myObject0.BStar,   32000, 1);
    myTree->Branch("S99_BStar",         &myObject0.BStar,   32000, 99);
    myTree->Branch("S101_BStar",            &myObject0.BStar,   32000, 101);

    // myTree->Branch("S0_BArray[12]",      "B[12]",        &myObject0.BArray,  32000, 0); // Will not get made
    // myTree->Branch(199_BArray[12]",  "B[12]",        &myObject0.BArray,  32000, 99); // Will not get made
    // myTree->Branch("S99_BArray[12]", "B[12]",        &myObject0.BArray,  32000, 99); // Will not get made

    // myTree->Branch("S0_BStarArray",      &myObject0.BStarArray,  32000, 0); // No way of specifying an array
    // myTree->Branch(199_BStarArray",  &myObject0.BStarArray,  32000, 99); // No way of specifying an array
    // myTree->Branch("S99_BStarArray", &myObject0.BStarArray,  32000, 99); // No way of specifying an array

    myTree->Branch("S0_BObject.",       &myObject0.BObject, 32000, 0);
    myTree->Branch("S1_BObject.",       &myObject0.BObject, 32000, 1);
    myTree->Branch("S99_BObject.",      &myObject0.BObject, 32000, 99);
    myTree->Branch("S101_BObject.",     &myObject0.BObject, 32000, 101);

    myTree->Branch("S0_BClonesArray",   &myObject0.BClonesArray,    32000, 0);
    myTree->Branch("S1_BClonesArray",   &myObject0.BClonesArray,    32000, 1);
    myTree->Branch("S99_BClonesArray",  &myObject0.BClonesArray,    32000, 99);
    myTree->Branch("S101_BClonesArray", &myObject0.BClonesArray,    32000, 101);

    myTree->Branch("MyLeafList",        &myLeaves,  "x:y/F:n/I:b[12]/O:a[n]/D");


    for (int i = startI; i < TREE_ENTRIES + startI; ++i){
        fprintf(stderr, "\nEntry %i\n\n", i);

        // Clear old values
        myObject0.ResetVectorFloat();
        myObject0.ResetVectorB();

        fprintf(stderr, "Setting BObject\n");
        myObject0.BObject.dummy = i;

        for (int j = 0; j < LIST_ENTRIES; ++j){
            // Vector of floats
            fprintf(stderr, "Adding %.2f to vectorFloat\n", (Float_t)(i*j));
            myObject0.AddToVectorFloat(i*j);

            // Vector of objects
            B obj (i*j);
            fprintf(stderr, "Adding %i to vectorB\n", i*j);
            myObject0.AddToVectorB(obj);

            obj.dummy *= 2;
            // Hangs makeTree
            fprintf(stderr, "Adding %i to vectorStarB\n",  obj.dummy);
            myObject0.AddToVectorStarB(obj);

            // Vector of pointers
            fprintf(stderr, "Adding %i to vectorBStar\n", i*j*2);
            B *ptr = new B(i*j*3);
            myObject0.AddToVectorBStar(ptr);
        }

        fprintf(stderr, "Setting BStar\n");
        B *objPtr = new B(i*2);
        myObject0.SetBStar(objPtr);

        fprintf(stderr, "Filling BArray\n");
        B *myArray = new B[12];
        for (int j = 0; j < 12; ++j){
            myArray[j].dummy = i * j * 4;
        }
        myObject0.FillBArray(myArray);
        delete [] myArray;

        fprintf(stderr, "Filling BStarArray\n");
        for (int j = 0; j < NUM_CONSTANT; ++j){
            myObject0.BStarArray[j] = i*j*5;
        }

        fprintf(stderr, "Filling BClonesArray\n");
        for (int j = 0; j < LIST_ENTRIES; ++j ){
            ((B*)myObject0.BClonesArray.New(j))->dummy = i*j*6;
        }

        fprintf(stderr, "Filling leaflist\n");
        myLeaves.myFloatX = (Float_t)i;
        myLeaves.myFloatY = (Float_t)i / 10.0f;
        for (int j = 0; j < MYDOUBLEARRAY_SIZE; ++j){
            myLeaves.myDoubleArrayA[j] = myLeaves.myFloatY * j;
        }
        for (int j = 0; j < MYBOOLARRAYB_SIZE; ++j){
            //myLeaves.myBoolArrayB[j] = (i + (i * j)) % 2;
            myLeaves.myBoolArrayB[j] = j % 2;
            //myLeaves.myBoolArrayB[j] = true;
        }

        fprintf(stderr, "Filling tree\n");
        myTree->Fill();

        myObject0.ResetVectorStarB();

        myObject0.vectorBStar.clear();

        fprintf(stderr, "Entry created\n");
    }
    fprintf(stderr, "Tree created\n");

    myFile->Write();
    delete myFile;
}

// Extracts either a single tree or creates a TChain from  the input files.
class TreeGetter {
   std::vector<std::string> fFileNames;
   TFile* fFile;
   TChain* fChain;

public:
   TreeGetter(): fFile(0), fChain(0) {}
   ~TreeGetter() { delete fFile; delete fChain; }
   void Add(const char* fileName) { fFileNames.push_back(fileName); }
   TTree* GetTree() {
      // If we have opened the file, close it.
      delete fFile;
      fFile = 0;
      // If we have created a TChain, get rid of it.
      delete fChain;
      fChain = 0;
      if (fFileNames.size() == 1) {
         // Single tree case.
         fFile = TFile::Open(fFileNames[0].c_str());
         TTree* tree = 0;
         fFile->GetObject("HardTree", tree);
         return tree;
      } else if (fFileNames.size() > 1) {
         // Multiple files, thus return a chain.
         fChain = new TChain("HardTree");
         for (const auto& str: fFileNames) {
            fChain->Add(str.c_str());
         }
         return fChain;
      }
      return 0;
   }
};

void readNum(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){
    TTreeReader myTreeReader (getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "num";

    TTreeReaderValue<Int_t> myNum (myTreeReader, finalBranchName);

    // Bool_t success = !myNum.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (testValues && *myNum != NUM_CONSTANT) success = false;
        if (printOut) fprintf(stderr, "Num: %i\n", *myNum);
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readBObject(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter) {
    TTreeReader myTreeReader (getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "BObject";

    TTreeReaderValue<B> myBObject (myTreeReader, finalBranchName);

    // Bool_t success = !myBObject.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (testValues && myBObject->dummy != i * MULTIPLIER_B_OBJECT) success = false;
        if (printOut) fprintf(stderr, "Dummy: %i\n", myBObject->dummy);
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readBObjectBranch(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){
    TTreeReader myTreeReader (getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "BObject.";

    TTreeReaderValue<B> myBObject (myTreeReader, finalBranchName);

    // Bool_t success = !myBObject.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (testValues && myBObject->dummy != i * MULTIPLIER_B_OBJECT) success = false;
        if (printOut) fprintf(stderr, "Dummy: %i\n", myBObject->dummy);
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readBObjectDummy(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "BObject.dummy";

    TTreeReaderValue<Int_t> myDummy (myTreeReader, finalBranchName);

    // Bool_t success = !myDummy.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (testValues && *myDummy != i * MULTIPLIER_B_OBJECT) success = false;
        if (printOut) fprintf(stderr, "Dummy: %i\n", *myDummy);
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readBStar(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "BStar";

    TTreeReaderValue<B> myBStar (myTreeReader, finalBranchName);

    // Bool_t success = !myBStar.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (testValues && myBStar->dummy != i * MULTIPLIER_B_STAR) success = false;
        if (printOut) fprintf(stderr, "Dummy: %i\n", myBStar->dummy);
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readVectorFloatValue(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "vectorFloat";

    TTreeReaderValue<std::vector<Float_t> > myVectorFloat (myTreeReader, finalBranchName);

    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "vectorFloat values:");

        for (int j = 0; j < LIST_ENTRIES; ++j){
            if (testValues && fabs(myVectorFloat->at(j) - i * j * MULTIPLIER_VECTOR_FLOAT) > 0.001f) success = false;
            if (printOut) fprintf(stderr, " %.2f", myVectorFloat->at(j));
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readVectorBValue(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "vectorB";

    TTreeReaderValue<std::vector<B> > myVectorB (myTreeReader, finalBranchName);

    // Bool_t success = !myVectorB.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "vectorB dummies:");

        for (int j = 0; j < LIST_ENTRIES; ++j){
            if (testValues && myVectorB->at(j).dummy != i * j * MULTIPLIER_VECTOR_B) success = false;
            if (printOut) fprintf(stderr, " %i", myVectorB->at(j).dummy);
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readVectorStarBValue(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "vectorStarB";

    TTreeReaderValue<std::vector<B> > myVectorStarB (myTreeReader, finalBranchName);

    // Bool_t success = !myVectorStarB.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "vectorB dummies(%lu):", myVectorStarB->size());

        for (int j = 0; j < LIST_ENTRIES; ++j){
            if (testValues && myVectorStarB->at(j).dummy != i * j * MULTIPLIER_VECTOR_STAR_B) success = false;
            if (printOut) fprintf(stderr, " %i", myVectorStarB->at(j).dummy);
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readVectorStarBArray(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "vectorStarB";

    TTreeReaderArray<B> myVectorStarB (myTreeReader, finalBranchName);

    // Bool_t success = !myVectorStarB.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "vectorB dummies(%lu):", myVectorStarB.GetSize());

        for (int j = 0; j < LIST_ENTRIES; ++j){
            if (testValues && myVectorStarB.At(j).dummy != i * j * MULTIPLIER_VECTOR_STAR_B) success = false;
            if (printOut) fprintf(stderr, " %i", myVectorStarB.At(j).dummy);
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readVectorFloatArray(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "vectorFloat";

    TTreeReaderArray<Float_t> myVectorFloat (myTreeReader, finalBranchName);

    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "vectorFloat values(%lu):", myVectorFloat.GetSize());

        for (int j = 0; j < LIST_ENTRIES && j < (int)myVectorFloat.GetSize(); ++j){
            if (testValues && fabs(myVectorFloat.At(j) - i * j * MULTIPLIER_VECTOR_FLOAT) > 0.001f) success = false;
            if (printOut) fprintf(stderr, " %.2f", myVectorFloat.At(j));
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readVectorBArray(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "vectorB";

    TTreeReaderArray<B> myVectorB (myTreeReader, finalBranchName);

    // Bool_t success = !myVectorB.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "vectorB dummies(%lu):", myVectorB.GetSize());

        for (int j = 0; j < LIST_ENTRIES && j < (int)myVectorB.GetSize(); ++j){
            if (testValues && myVectorB.At(j).dummy != i * j * MULTIPLIER_VECTOR_B) success = false;
            if (printOut) fprintf(stderr, " %i", myVectorB.At(j).dummy);
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readBArray(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "BArray[12]";

    TTreeReaderArray<B> myBArray (myTreeReader, finalBranchName);

    // Bool_t success = !myBArray.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "BArray dummies(%lu):", myBArray.GetSize());

        for (int j = 0; j < LIST_ENTRIES; ++j){
            if (testValues && myBArray.At(j).dummy != i * j * MULTIPLIER_B_ARRAY) success = false;
            if (printOut) fprintf(stderr, " %i", myBArray.At(j).dummy);
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readBStarArray(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "BStarArray";

    TTreeReaderArray<B> myBStarArray (myTreeReader, finalBranchName);

    // Bool_t success = !myBStarArray.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "BStarArray dummies(%lu):", myBStarArray.GetSize());

        for (int j = 0; j < (int)myBStarArray.GetSize(); ++j){
            if (testValues && myBStarArray.At(j).dummy != i * j * MULTIPLIER_B_STAR_ARRAY) success = false;
            if (printOut) fprintf(stderr, " %i", myBStarArray.At(j).dummy);
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readVectorBStarValue(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "vectorBStar";

    TTreeReaderValue<std::vector<B*> > myVectorBStar (myTreeReader, finalBranchName);

    // Bool_t success = !myVectorBStar.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "vectorBStar dummies(%lu):", myVectorBStar->size());

        for (int j = 0; j < LIST_ENTRIES && j < (int)myVectorBStar->size(); ++j){
            if (testValues && myVectorBStar->at(j)->dummy != i * j * MULTIPLIER_VECTOR_B_STAR) success = false;
            if (printOut) fprintf(stderr, " %i", myVectorBStar->at(j)->dummy);
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readVectorBStarArray(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "vectorBStar";

    TTreeReaderArray<B> myVectorBStar (myTreeReader, finalBranchName);

    // Bool_t success = !myVectorBStar.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "vectorBStar dummies(%lu):", myVectorBStar.GetSize());

        for (int j = 0; j < LIST_ENTRIES && (int)myVectorBStar.GetSize(); ++j){
            if (testValues && myVectorBStar.At(j).dummy != i * j * MULTIPLIER_VECTOR_B_STAR) success = false;
            if (printOut) fprintf(stderr, " %i", myVectorBStar.At(j).dummy);
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readBClonesArrayValue(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "BClonesArray";

    TTreeReaderValue<TClonesArray> myBClonesArray (myTreeReader, finalBranchName);

    // Bool_t success = !myBClonesArray.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "BClonesArray dummies(%i):", myBClonesArray->GetEntries());

        for (int j = 0; j < LIST_ENTRIES && j < myBClonesArray->GetEntries(); ++j){
            if (testValues && ((B*)myBClonesArray->At(j))->dummy != i * j * MULTIPLIER_B_CLONES_ARRAY) success = false;
            if (printOut) fprintf(stderr, " %i", ((B*)myBClonesArray->At(j))->dummy);
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readBClonesArrayArray(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "BClonesArray";

    TTreeReaderArray<B> myBClonesArray (myTreeReader, finalBranchName);

    // Bool_t success = !myBClonesArray.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "BClonesArray dummies(%lu):", myBClonesArray.GetSize());

        for (int j = 0; j < LIST_ENTRIES && j < (int)myBClonesArray.GetSize(); ++j){
            if (testValues && myBClonesArray.At(j).dummy != i * j * MULTIPLIER_B_CLONES_ARRAY) success = false;
            if (printOut) fprintf(stderr, " %i", myBClonesArray.At(j).dummy);
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readVectorBDummyArray(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "vectorB.dummy";

    TTreeReaderArray<Int_t> myVectorBDummyArray (myTreeReader, finalBranchName);

    // Bool_t success = !myVectorBDummyArray.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "vectorB.dummies(%lu):", myVectorBDummyArray.GetSize());

        for (int j = 0; j < LIST_ENTRIES && j < (int)myVectorBDummyArray.GetSize(); ++j){
            if (testValues && myVectorBDummyArray.At(j) != i * j * MULTIPLIER_VECTOR_B) success = false;
            if (printOut) fprintf(stderr, " %i", myVectorBDummyArray.At(j));
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readBClonesArrayDummyArray(const char* branchName, Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString finalBranchName = branchName;
    finalBranchName += "BClonesArray.dummy";

    TTreeReaderArray<Int_t> myBClonesArrayDummy (myTreeReader, finalBranchName);

    // Bool_t success = !myBClonesArrayDummy.GetSetupStatus();
    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "BClonesArray.dummies(%lu):", myBClonesArrayDummy.GetSize());

        for (int j = 0; j < LIST_ENTRIES && j < (int)myBClonesArrayDummy.GetSize(); ++j){
            if (testValues && myBClonesArrayDummy.At(j) != i * j * MULTIPLIER_B_CLONES_ARRAY) success = false;
            if (printOut) fprintf(stderr, " %i", myBClonesArrayDummy.At(j));
        }

        if (printOut) fprintf(stderr, "\n");
    }
    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readLeafFloatX(Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString branchName = "MyLeafList.x";

    TTreeReaderValue<Float_t> myFloat (myTreeReader, branchName);

    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (testValues && *myFloat - (Float_t)i > 0.0001f) success = false;
        if (printOut) fprintf(stderr, "MyLeafList.x: %f\n", *myFloat);
    }

    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readLeafFloatY(Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString branchName = "MyLeafList.y";

    TTreeReaderValue<Float_t> myFloat (myTreeReader, branchName);

    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (testValues && *myFloat - ((Float_t)i / 10.0f) > 0.0001f) success = false;
        if (printOut) fprintf(stderr, "MyLeafList.y: %f\n", *myFloat);
    }

    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readLeafIntN(Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString branchName = "MyLeafList.n";

    TTreeReaderValue<Int_t> myInt (myTreeReader, branchName);

    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (testValues && *myInt != MYDOUBLEARRAY_SIZE) success = false;
        if (printOut) fprintf(stderr, "MyLeafList.n: %i\n", *myInt);
    }

    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readLeafDoubleAArray(Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString branchName = "MyLeafList.a";

    TTreeReaderArray<Double_t> myDoubles (myTreeReader, branchName);

    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "MyLeafList.a(%lu):", myDoubles.GetSize());

        for (size_t j = 0; j < myDoubles.GetSize() && j < 10; ++j){
            if (testValues && fabs(myDoubles.At(j) - (i * j) / 10.0f) > 0.0001f) success = false;
            if (printOut) fprintf(stderr, " %f", myDoubles.At(j));
        }

        if (printOut) fprintf(stderr, "\n");
    }

    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readLeafBoolBArray(Bool_t printOut, Bool_t testValues, TreeGetter& getter){

    TTreeReader myTreeReader(getter.GetTree());

    TString branchName = "MyLeafList.b";

    TTreeReaderArray<Bool_t> myBools (myTreeReader, branchName);

    Bool_t success = true;
    Bool_t read = false;
    for (int i = 1; myTreeReader.Next(); ++i){
        read = true;
        if (printOut) fprintf(stderr, "MyLeafList.b(%lu):", myBools.GetSize());

        for (size_t j = 0; j < myBools.GetSize() && j < 10; ++j){
            if (testValues && myBools.At(j) != j % 2) success = false;
            if (printOut) fprintf(stderr, " %s", myBools.At(j) ? "true" : "false" );
        }

        if (printOut) fprintf(stderr, "\n");
    }

    if (testValues) fprintf(stderr, "%s\n", success && read ? "Success!" : "Failure");
}

void readTree(const char* fileName = "HardTreeFile.root"){
    TFile *myFile = TFile::Open(fileName);
    TTree *myTree = (TTree*)myFile->Get("HardTree");
    myTree->Print();

    for (int i = 0; i < 10 && i < myTree->GetEntries(); ++i){
        myTree->Show(i);
    }
    delete myFile;
}


void output(Bool_t printAll, Bool_t testAll, TreeGetter& getter){
    fprintf(stderr, "A0: readNum(): ----------------------------- %s", printAll ? "\n": ""); readNum(                    "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readBObject(): ------------------------- %s", printAll ? "\n": ""); readBObject(                "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readBObjectDummy(): -------------------- %s", printAll ? "\n": ""); readBObjectDummy(           "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readBStar(): --------------------------- %s", printAll ? "\n": ""); readBStar(                  "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readVectorFloatValue(): ---------------- %s", printAll ? "\n": ""); readVectorFloatValue(       "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readVectorBValue(): -------------------- %s", printAll ? "\n": ""); readVectorBValue(           "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readVectorStarBValue(): ---------------- %s", printAll ? "\n": ""); readVectorStarBValue(       "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readVectorStarBArray(): ---------------- %s", printAll ? "\n": ""); readVectorStarBArray(       "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readVectorFloatArray(): ---------------- %s", printAll ? "\n": ""); readVectorFloatArray(       "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readVectorBArray(): -------------------- %s", printAll ? "\n": ""); readVectorBArray(           "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readBArray(): -------------------------- %s", printAll ? "\n": ""); readBArray(                 "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readBStarArray(): ---------------------- %s", printAll ? "\n": ""); readBStarArray(             "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readVectorBStarValue(): ---------------- %s", printAll ? "\n": ""); readVectorBStarValue(       "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readVectorBStarArray(): ---------------- %s", printAll ? "\n": ""); readVectorBStarArray(       "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readBClonesArrayValue(): --------------- %s", printAll ? "\n": ""); readBClonesArrayValue(      "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readBClonesArrayArray(): --------------- %s", printAll ? "\n": ""); readBClonesArrayArray(      "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readVectorBDummyArray(): --------------- %s", printAll ? "\n": ""); readVectorBDummyArray(      "A0.", printAll, testAll, getter);
    fprintf(stderr, "A0: readBClonesArrayDummyArray(): ---------- %s", printAll ? "\n": ""); readBClonesArrayDummyArray( "A0.", printAll, testAll, getter);

    fprintf(stderr, "A1: readNum(): ------------------------------ %s", printAll ? "\n": ""); readNum(                   "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readBObject(): -------------------------- %s", printAll ? "\n": ""); readBObject(               "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readBStar(): ---------------------------- %s", printAll ? "\n": ""); readBStar(                 "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readVectorFloatValue(): ----------------- %s", printAll ? "\n": ""); readVectorFloatValue(      "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readVectorBValue(): --------------------- %s", printAll ? "\n": ""); readVectorBValue(          "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readVectorStarBValue(): ----------------- %s", printAll ? "\n": ""); readVectorStarBValue(      "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readVectorStarBArray(): ----------------- %s", printAll ? "\n": ""); readVectorStarBArray(      "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readVectorFloatArray(): ----------------- %s", printAll ? "\n": ""); readVectorFloatArray(      "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readVectorBArray(): --------------------- %s", printAll ? "\n": ""); readVectorBArray(          "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readBArray(): --------------------------- %s", printAll ? "\n": ""); readBArray(                "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readBStarArray(): ----------------------- %s", printAll ? "\n": ""); readBStarArray(            "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readVectorBStarValue(): ----------------- %s", printAll ? "\n": ""); readVectorBStarValue(      "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readVectorBStarArray(): ----------------- %s", printAll ? "\n": ""); readVectorBStarArray(      "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readBClonesArrayValue(): ---------------- %s", printAll ? "\n": ""); readBClonesArrayValue(     "A1.", printAll, testAll, getter);
    fprintf(stderr, "A1: readBClonesArrayArray(): ---------------- %s", printAll ? "\n": ""); readBClonesArrayArray(     "A1.", printAll, testAll, getter);


    fprintf(stderr, "A99: readNum(): ----------------------------- %s", printAll ? "\n": ""); readNum(                   "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readBObject(): ------------------------- %s", printAll ? "\n": ""); readBObject(               "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readBObjectDummy(): -------------------- %s", printAll ? "\n": ""); readBObjectDummy(          "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readBStar(): --------------------------- %s", printAll ? "\n": ""); readBStar(                 "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readVectorFloatValue(): ---------------- %s", printAll ? "\n": ""); readVectorFloatValue(      "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readVectorBValue(): -------------------- %s", printAll ? "\n": ""); readVectorBValue(          "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readVectorStarBValue(): ---------------- %s", printAll ? "\n": ""); readVectorStarBValue(      "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readVectorStarBArray(): ---------------- %s", printAll ? "\n": ""); readVectorStarBArray(      "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readVectorFloatArray(): ---------------- %s", printAll ? "\n": ""); readVectorFloatArray(      "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readVectorBArray(): -------------------- %s", printAll ? "\n": ""); readVectorBArray(          "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readBArray(): -------------------------- %s", printAll ? "\n": ""); readBArray(                "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readBStarArray(): ---------------------- %s", printAll ? "\n": ""); readBStarArray(            "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readVectorBStarValue(): ---------------- %s", printAll ? "\n": ""); readVectorBStarValue(      "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readVectorBStarArray(): ---------------- %s", printAll ? "\n": ""); readVectorBStarArray(      "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readBClonesArrayValue(): --------------- %s", printAll ? "\n": ""); readBClonesArrayValue(     "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readBClonesArrayArray(): --------------- %s", printAll ? "\n": ""); readBClonesArrayArray(     "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readVectorBDummyArray(): --------------- %s", printAll ? "\n": ""); readVectorBDummyArray(     "A99.", printAll, testAll, getter);
    fprintf(stderr, "A99: readBClonesArrayDummyArray(): ---------- %s", printAll ? "\n": ""); readBClonesArrayDummyArray("A99.", printAll, testAll, getter);
    //fprintf(stderr, "readAObject(): ------------------------ %s", printAll ? "\n": ""); readAObject();


    fprintf(stderr, "A101: readNum(): ---------------------------- %s", printAll ? "\n": ""); readNum(                   "A101.", printAll, testAll, getter);
    fprintf(stderr, "A101: readBObject(): ------------------------ %s", printAll ? "\n": ""); readBObject(               "A101.", printAll, testAll, getter);
    fprintf(stderr, "A101: readBStar(): -------------------------- %s", printAll ? "\n": ""); readBStar(                 "A101.", printAll, testAll, getter);
    fprintf(stderr, "A101: readVectorBValue(): ------------------- %s", printAll ? "\n": ""); readVectorBValue(          "A101.", printAll, testAll, getter);
    fprintf(stderr, "A101: readVectorStarBValue(): --------------- %s", printAll ? "\n": ""); readVectorStarBValue(      "A101.", printAll, testAll, getter);
    fprintf(stderr, "A101: readVectorStarBArray(): --------------- %s", printAll ? "\n": ""); readVectorStarBArray(      "A101.", printAll, testAll, getter);
    fprintf(stderr, "A101: readVectorBArray(): ------------------- %s", printAll ? "\n": ""); readVectorBArray(          "A101.", printAll, testAll, getter);
    fprintf(stderr, "A101: readBArray(): ------------------------- %s", printAll ? "\n": ""); readBArray(                "A101.", printAll, testAll, getter);
    fprintf(stderr, "A101: readBStarArray(): --------------------- %s", printAll ? "\n": ""); readBStarArray(            "A101.", printAll, testAll, getter);
    //fprintf(stderr, "A101: readVectorBStarValue(): --------------- %s", printAll ? "\n": ""); readVectorBStarValue(            "A101.", printAll, testAll, getter); // Segfault
    //fprintf(stderr, "A101: readVectorBStarArray(): --------------- %s", printAll ? "\n": ""); readVectorBStarArray(            "A101.", printAll, testAll, getter); // Segfault
    fprintf(stderr, "A101: readBClonesArrayValue(): -------------- %s", printAll ? "\n": ""); readBClonesArrayValue(     "A101.", printAll, testAll, getter);
    fprintf(stderr, "A101: readBClonesArrayArray(): -------------- %s", printAll ? "\n": ""); readBClonesArrayArray(     "A101.", printAll, testAll, getter);


    // fprintf(stderr, "S0_: readNum(): ----------------------------- %s", printAll ? "\n": ""); readNum(                        "S0_", printAll, testAll, getter); // Leaflist
    fprintf(stderr, "S0_: readBObject(): ------------------------- %s", printAll ? "\n": ""); readBObjectBranch(                 "S0_", printAll, testAll, getter);
    // fprintf(stderr, "S0_: readBObjectDummy(): -------------------- %s", printAll ? "\n": ""); readBObjectDummy(               "S0_", printAll, testAll, getter); // Branch not created
    fprintf(stderr, "S0_: readBStar(): --------------------------- %s", printAll ? "\n": ""); readBStar(                 "S0_", printAll, testAll, getter);
    fprintf(stderr, "S0_: readVectorFloatValue(): ---------------- %s", printAll ? "\n": ""); readVectorFloatValue(          "S0_", printAll, testAll, getter);
    fprintf(stderr, "S0_: readVectorBValue(): -------------------- %s", printAll ? "\n": ""); readVectorBValue(              "S0_", printAll, testAll, getter);
    fprintf(stderr, "S0_: readVectorStarBValue(): ---------------- %s", printAll ? "\n": ""); readVectorStarBValue(          "S0_", printAll, testAll, getter);
    fprintf(stderr, "S0_: readVectorStarBArray(): ---------------- %s", printAll ? "\n": ""); readVectorStarBArray(          "S0_", printAll, testAll, getter);
    fprintf(stderr, "S0_: readVectorFloatArray(): ---------------- %s", printAll ? "\n": ""); readVectorFloatArray(          "S0_", printAll, testAll, getter);
    fprintf(stderr, "S0_: readVectorBArray(): -------------------- %s", printAll ? "\n": ""); readVectorBArray(              "S0_", printAll, testAll, getter);
    // fprintf(stderr, "S0_: readBArray(): -------------------------- %s", printAll ? "\n": ""); readBArray(                 "S0_", printAll, testAll, getter); // Branch not created
    // fprintf(stderr, "S0_: readBStarArray(): ---------------------- %s", printAll ? "\n": ""); readBStarArray(             "S0_", printAll, testAll, getter); // Branch not created
    fprintf(stderr, "S0_: readVectorBStarValue(): ---------------- %s", printAll ? "\n": ""); readVectorBStarValue(          "S0_", printAll, testAll, getter);
    fprintf(stderr, "S0_: readVectorBStarArray(): ---------------- %s", printAll ? "\n": ""); readVectorBStarArray(          "S0_", printAll, testAll, getter);
    // fprintf(stderr, "S0_: readBClonesArrayValue(): --------------- %s", printAll ? "\n": ""); readBClonesArrayValue(      "S0_", printAll, testAll, getter); // TBranchProxy->Read() fails
    // fprintf(stderr, "S0_: readBClonesArrayArray(): --------------- %s", printAll ? "\n": ""); readBClonesArrayArray(      "S0_", printAll, testAll, getter); // TBranchProxy->Read() fails
    // fprintf(stderr, "S0_: readVectorBDummyArray(): --------------- %s", printAll ? "\n": ""); readVectorBDummyArray(      "S0_", printAll, testAll, getter); // Branch not found
    // fprintf(stderr, "S0_: readBClonesArrayDummyArray(): ---------- %s", printAll ? "\n": ""); readBClonesArrayDummyArray( "S0_", printAll, testAll, getter); // Branch not found


    // fprintf(stderr, "S1_: readNum(): ----------------------------- %s", printAll ? "\n": ""); readNum(                        "S1_", printAll, testAll, getter); // Leaflist
    fprintf(stderr, "S1_: readBObject(): ------------------------- %s", printAll ? "\n": ""); readBObjectBranch(                 "S1_", printAll, testAll, getter);
    fprintf(stderr, "S1_: readBObjectDummy(): -------------------- %s", printAll ? "\n": ""); readBObjectDummy(              "S1_", printAll, testAll, getter);
    fprintf(stderr, "S1_: readBStar(): --------------------------- %s", printAll ? "\n": ""); readBStar(                 "S1_", printAll, testAll, getter);
    fprintf(stderr, "S1_: readVectorBValue(): -------------------- %s", printAll ? "\n": ""); readVectorBValue(              "S1_", printAll, testAll, getter);
    fprintf(stderr, "S1_: readVectorStarBValue(): ---------------- %s", printAll ? "\n": ""); readVectorStarBValue(          "S1_", printAll, testAll, getter);
    fprintf(stderr, "S1_: readVectorStarBArray(): ---------------- %s", printAll ? "\n": ""); readVectorStarBArray(          "S1_", printAll, testAll, getter);
    fprintf(stderr, "S1_: readVectorBArray(): -------------------- %s", printAll ? "\n": ""); readVectorBArray(              "S1_", printAll, testAll, getter);
    // fprintf(stderr, "S1_: readBArray(): -------------------------- %s", printAll ? "\n": ""); readBArray(                 "S1_", printAll, testAll, getter); // Branch not created
    // fprintf(stderr, "S1_: readBStarArray(): ---------------------- %s", printAll ? "\n": ""); readBStarArray(             "S1_", printAll, testAll, getter); // Branch not created
    fprintf(stderr, "S1_: readVectorBStarValue(): ---------------- %s", printAll ? "\n": ""); readVectorBStarValue(          "S1_", printAll, testAll, getter);
    fprintf(stderr, "S1_: readVectorBStarArray(): ---------------- %s", printAll ? "\n": ""); readVectorBStarArray(          "S1_", printAll, testAll, getter);
    fprintf(stderr, "S1_: readBClonesArrayValue(): --------------- %s", printAll ? "\n": ""); readBClonesArrayValue(     "S1_", printAll, testAll, getter);
    fprintf(stderr, "S1_: readBClonesArrayArray(): --------------- %s", printAll ? "\n": ""); readBClonesArrayArray(     "S1_", printAll, testAll, getter);
    fprintf(stderr, "S1_: readVectorBDummyArray(): --------------- %s", printAll ? "\n": ""); readVectorBDummyArray(     "S1_", printAll, testAll, getter);
    fprintf(stderr, "S1_: readBClonesArrayDummyArray(): ---------- %s", printAll ? "\n": ""); readBClonesArrayDummyArray(    "S1_", printAll, testAll, getter);


    // fprintf(stderr, "S99_: readNum(): ---------------------------- %s", printAll ? "\n": ""); readNum(                        "S99_", printAll, testAll, getter); // Leaflist
    fprintf(stderr, "S99_: readBObject(): ------------------------ %s", printAll ? "\n": ""); readBObjectBranch(                 "S99_", printAll, testAll, getter);
    fprintf(stderr, "S99_: readBObjectDummy(): ------------------- %s", printAll ? "\n": ""); readBObjectDummy(              "S99_", printAll, testAll, getter);
    fprintf(stderr, "S99_: readBStar(): -------------------------- %s", printAll ? "\n": ""); readBStar(                 "S99_", printAll, testAll, getter);
    fprintf(stderr, "S99_: readVectorBValue(): ------------------- %s", printAll ? "\n": ""); readVectorBValue(              "S99_", printAll, testAll, getter);
    fprintf(stderr, "S99_: readVectorStarBValue(): --------------- %s", printAll ? "\n": ""); readVectorStarBValue(          "S99_", printAll, testAll, getter);
    fprintf(stderr, "S99_: readVectorStarBArray(): --------------- %s", printAll ? "\n": ""); readVectorStarBArray(          "S99_", printAll, testAll, getter);
    fprintf(stderr, "S99_: readVectorBArray(): ------------------- %s", printAll ? "\n": ""); readVectorBArray(              "S99_", printAll, testAll, getter);
    // fprintf(stderr, "S99_: readBArray(): ------------------------- %s", printAll ? "\n": ""); readBArray(                 "S99_", printAll, testAll, getter); // Branch not created
    // fprintf(stderr, "S99_: readBStarArray(): --------------------- %s", printAll ? "\n": ""); readBStarArray(             "S99_", printAll, testAll, getter); // Branch not created
    fprintf(stderr, "S99_: readVectorBStarValue(): --------------- %s", printAll ? "\n": ""); readVectorBStarValue(          "S99_", printAll, testAll, getter);
    fprintf(stderr, "S99_: readVectorBStarArray(): --------------- %s", printAll ? "\n": ""); readVectorBStarArray(          "S99_", printAll, testAll, getter);
    fprintf(stderr, "S99_: readBClonesArrayValue(): -------------- %s", printAll ? "\n": ""); readBClonesArrayValue(     "S99_", printAll, testAll, getter);
    fprintf(stderr, "S99_: readBClonesArrayArray(): -------------- %s", printAll ? "\n": ""); readBClonesArrayArray(     "S99_", printAll, testAll, getter);
    fprintf(stderr, "S99_: readVectorBDummyArray(): -------------- %s", printAll ? "\n": ""); readVectorBDummyArray(     "S99_", printAll, testAll, getter);
    fprintf(stderr, "S99_: readBClonesArrayDummyArray(): --------- %s", printAll ? "\n": ""); readBClonesArrayDummyArray(    "S99_", printAll, testAll, getter);

    // fprintf(stderr, "S101_: readNum(): --------------------------- %s", printAll ? "\n": ""); readNum(                        "S101_", printAll, testAll, getter); // Leaflist
    fprintf(stderr, "S101_: readBObject(): ----------------------- %s", printAll ? "\n": ""); readBObjectBranch(                 "S101_", printAll, testAll, getter);
    fprintf(stderr, "S101_: readBObjectDummy(): ------------------ %s", printAll ? "\n": ""); readBObjectDummy(              "S101_", printAll, testAll, getter);
    fprintf(stderr, "S101_: readBStar(): ------------------------- %s", printAll ? "\n": ""); readBStar(                 "S101_", printAll, testAll, getter);
    // fprintf(stderr, "S101_: readVectorBValue(): ------------------ %s", printAll ? "\n": ""); readVectorBValue(               "S101_", printAll, testAll, getter); // Branch not created
    // fprintf(stderr, "S101_: readVectorStarBValue(): -------------- %s", printAll ? "\n": ""); readVectorStarBValue(           "S101_", printAll, testAll, getter); // Branch not created
    // fprintf(stderr, "S101_: readVectorStarBArray(): -------------- %s", printAll ? "\n": ""); readVectorStarBArray(           "S101_", printAll, testAll, getter); // Branch not created
    // fprintf(stderr, "S101_: readVectorBArray(): ------------------ %s", printAll ? "\n": ""); readVectorBArray(               "S101_", printAll, testAll, getter); // Branch not created
    // fprintf(stderr, "S101_: readBArray(): ------------------------ %s", printAll ? "\n": ""); readBArray(                 "S101_", printAll, testAll, getter); // Branch not created
    // fprintf(stderr, "S101_: readBStarArray(): -------------------- %s", printAll ? "\n": ""); readBStarArray(             "S101_", printAll, testAll, getter); // Branch not created
    // fprintf(stderr, "S101_: readVectorBStarValue(): -------------- %s", printAll ? "\n": ""); readVectorBStarValue(           "S101_", printAll, testAll, getter); // Branch not created
    // fprintf(stderr, "S101_: readVectorBStarArray(): -------------- %s", printAll ? "\n": ""); readVectorBStarArray(           "S101_", printAll, testAll, getter); // Branch not created
    fprintf(stderr, "S101_: readBClonesArrayValue(): ------------- %s", printAll ? "\n": ""); readBClonesArrayValue(     "S101_", printAll, testAll, getter);
    fprintf(stderr, "S101_: readBClonesArrayArray(): ------------- %s", printAll ? "\n": ""); readBClonesArrayArray(     "S101_", printAll, testAll, getter);
    // fprintf(stderr, "S101_: readVectorBDummyArray(): ------------- %s", printAll ? "\n": ""); readVectorBDummyArray(      "S101_", printAll, testAll, getter);  // Branch not created
    fprintf(stderr, "S101_: readBClonesArrayDummyArray(): -------- %s", printAll ? "\n": ""); readBClonesArrayDummyArray(    "S101_", printAll, testAll, getter);

    fprintf(stderr, "readLeafFloatX(): --------------------------- %s", printAll ? "\n": ""); readLeafFloatX(        printAll, testAll, getter);
    fprintf(stderr, "readLeafFloatY(): --------------------------- %s", printAll ? "\n": ""); readLeafFloatY(        printAll, testAll, getter);
    fprintf(stderr, "readLeafIntN(): ----------------------------- %s", printAll ? "\n": ""); readLeafIntN(          printAll, testAll, getter);
    fprintf(stderr, "readLeafDoubleAArray(): --------------------- %s", printAll ? "\n": ""); readLeafDoubleAArray(  printAll, testAll, getter);
    fprintf(stderr, "readLeafBoolBArray(): ----------------------- %s", printAll ? "\n": ""); readLeafBoolBArray(    printAll, testAll, getter);
}

void testAll(){
   TreeGetter tg;
   tg.Add("./HardTreeFile.root");
   output(false, true, tg);
}

void printAll(){
   TreeGetter tg;
   tg.Add("./HardTreeFile.root");
   output(true, true, tg);
}

void testChain(){
   TreeGetter tg;
   tg.Add("./HardTreeFile.root");
   tg.Add("./HardTreeFile2.root");
   output(false, true, tg);
}

void printChain(){
   TreeGetter tg;
   tg.Add("./HardTreeFile.root");
   tg.Add("./HardTreeFile2.root");
   output(true, true, tg);
}
