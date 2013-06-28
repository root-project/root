#include "TFile.h"
#include "TTreeReader.h"
#include "TInterpreter.h"
#include "TSystem.h"
#include "MyParticle.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include <vector>
#include "A.h"
#include "B.h"

#ifdef __CINT__
#pragma link C++ class TTreeReaderValue<std::vector<MyParticle*>>+;
#pragma link C++ class TTreeReaderValue<MyParticle>+;
#pragma link C++ class TTreeReaderValue<Int_t>+;
#pragma link C++ class TTreeReaderValue<B>+;
#pragma link C++ class TTreeReaderValue<A>+;
#pragma link C++ class TTreeReaderValue<std::vector<B>>+;
#pragma link C++ class TTreeReaderValue<B*>+;
#pragma link C++ class TTreeReaderValue<TClonesArray>+;
#pragma link C++ class TTreeReaderValue<std::vector<B*>>+;
#pragma link C++ class TTreeReaderArray<double>+;
#pragma link C++ class TTreeReaderArray<B>+;
#pragma link C++ class TTreeReaderArray<Int_t>+;
#pragma link C++ class std::vector<B*>+;
#endif

#define TREE_ENTRIES 10
#define LIST_ENTRIES 10

#define NUM_CONSTANT 14
#define MULTIPLIER_B_OBJECT 1
#define MULTIPLIER_VECTOR_B 1
#define MULTIPLIER_VECTOR_B_STAR 3
#define MULTIPLIER_VECTOR_STAR_B 2
#define MULTIPLIER_B_STAR 2
#define MULTIPLIER_B_ARRAY 4
#define MULTIPLIER_B_STAR_ARRAY 5
#define MULTIPLIER_B_CLONES_ARRAY 6

void makeTree(){
	TFile *myFile = new TFile("HardTreeFile.root", "RECREATE");
	TTree *myTree = new TTree("HardTree", "This is hard to read");

	A myObject0 (NUM_CONSTANT);

	myTree->Branch("A0.",	"A", 	&myObject0,		32000, 0);
	myTree->Branch("A1.",	"A", 	&myObject0,		32000, 1);
	myTree->Branch("A99.",	"A", 	&myObject0,	32000, 99);
	myTree->Branch("A101.",	"A", 	&myObject0,	32000, 101);

	// myTree->Branch("vectorB_0.",	&branchForMembers.vectorB,	32000, 0);
	// myTree->Branch("vectorB_99.",	&branchForMembers.vectorB,	32000, 99);

	// myTree->Branch("vectorBStar_0.",	&branchForMembers.vectorBStar,	32000, 0);
	// myTree->Branch("vectorBStar_99.",	&branchForMembers.vectorBStar,	32000, 99);
	// myTree->Branch("vectorBStar_101.",	&branchForMembers.vectorBStar,	32000, 101);

	myTree->Branch("S0_num",			&myObject0.num,	32000, 0);
	myTree->Branch("S99_num",			&myObject0.num,	32000, 99);

	myTree->Branch("S0_vectorB",		&myObject0.vectorB,	32000, 0);
	myTree->Branch("S99_vectorB",		&myObject0.vectorB,	32000, 99);

	myTree->Branch("S0_vectorBStar",	&myObject0.vectorBStar,	32000, 0);
	myTree->Branch("S99_vectorBStar",	&myObject0.vectorBStar,	32000, 99);
	//myTree->Branch("vectorBStar_101",	&branchForMembers.vectorBStar,	32000, 101); // Breaks Fill()

	myTree->Branch("S0_vectorStarB",	&myObject0.vectorStarB,	32000, 0);
	myTree->Branch("S99_vectorStarB",	&myObject0.vectorStarB,	32000, 99);

	myTree->Branch("S0_BStar",			&myObject0.BStar,	32000, 0);
	myTree->Branch("S99_BStar",		&myObject0.BStar,	32000, 99);

	// myTree->Branch("S0_BArray[12]",		"B[12]",		&myObject0.BArray,	32000, 0); // Will not get made
	// myTree->Branch("S99_BArray[12]",	"B[12]",		&myObject0.BArray,	32000, 99); // Will not get made

	myTree->Branch("S0_BStarArray",	&myObject0.BStarArray,	32000, 0);
	myTree->Branch("S99_BStarArray",	&myObject0.BStarArray,	32000, 99);

	myTree->Branch("S0_BObject",		&myObject0.BObject,	32000, 0);
	myTree->Branch("S99_BObject",		&myObject0.BObject,	32000, 99);

	myTree->Branch("S0_BClonesArray",	&myObject0.BClonesArray,	32000, 0);
	myTree->Branch("S99_BClonesArray",	&myObject0.BClonesArray,	32000, 99);


	for (int i = 1; i < TREE_ENTRIES + 1; ++i){
		printf("\nEntry %i\n\n", i);

		// Clear old values
		myObject0.ResetVectorB();

		printf("Setting BObject\n");
		myObject0.BObject.dummy = i;

		for (int j = 0; j < LIST_ENTRIES; ++j){
			// Vector of objects
			B obj (i*j);
			printf("Adding %i to vectorB\n", i*j);
			myObject0.AddToVectorB(obj);

			obj.dummy *= 2;
			// Hangs makeTree
			printf("Adding %i to vectorStarB\n",  obj.dummy);
			myObject0.AddToVectorStarB(obj);

			// Vector of pointers
			printf("Adding %i to vectorBStar\n", i*j*2);
			B *ptr = new B(i*j*3);
			myObject0.AddToVectorBStar(ptr);
		}

		printf("Setting BStar\n");
		B *objPtr = new B(i*2);
		myObject0.SetBStar(objPtr);

		printf("Filling BArray\n");
		B *myArray = new B[12];
		for (int j = 0; j < 12; ++j){
			myArray[j].dummy = i * j * 4;
		}
		myObject0.FillBArray(myArray);
		delete [] myArray;

		printf("Filling BStarArray\n");
		for (int j = 0; j < NUM_CONSTANT; ++j){
			myObject0.BStarArray[j] = i*j*5;
		}

		printf("Filling BClonesArray\n");
		for (int j = 0; j < LIST_ENTRIES; ++j ){
			((B*)myObject0.BClonesArray.New(j))->dummy = i*j*6;
		}

		printf("Filling tree\n");
		myTree->Fill();

		myObject0.ResetVectorStarB();

		myObject0.vectorBStar.clear();

		printf("Entry created\n");
	}
	printf("Tree created\n");

	myFile->Write();
}

void readNum(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "num";

	TTreeReaderValue<Int_t> myNum (myTreeReader, finalBranchName);

	// Bool_t success = !myNum.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (testValues && *myNum != NUM_CONSTANT) success = false;
		if (printOut) printf("Num: %i\n", *myNum);
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readBObject(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "BObject";

	TTreeReaderValue<B> myBObject (myTreeReader, finalBranchName);

	// Bool_t success = !myBObject.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (testValues && myBObject->dummy != i * MULTIPLIER_B_OBJECT) success = false;
		if (printOut) printf("Dummy: %i\n", myBObject->dummy);
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readBObjectDummy(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "BObject.dummy";

	TTreeReaderValue<Int_t> myDummy (myTreeReader, finalBranchName);

	// Bool_t success = !myDummy.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (testValues && *myDummy != i * MULTIPLIER_B_OBJECT) success = false;
		if (printOut) printf("Dummy: %i\n", *myDummy);
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readBStar(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "BStar";

	TTreeReaderValue<B> myBStar (myTreeReader, finalBranchName);

	// Bool_t success = !myBStar.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (testValues && myBStar->dummy != i * MULTIPLIER_B_STAR) success = false;
		if (printOut) printf("Dummy: %i\n", myBStar->dummy);
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readVectorBValue(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "vectorB";

	TTreeReaderValue<std::vector<B> > myVectorB (myTreeReader, finalBranchName);

	// Bool_t success = !myVectorB.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("vectorB dummies:");

		for (int j = 0; j < LIST_ENTRIES; ++j){
			if (testValues && myVectorB->at(j).dummy != i * j * MULTIPLIER_VECTOR_B) success = false;
			if (printOut) printf(" %i", myVectorB->at(j).dummy);
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readVectorStarBValue(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "vectorStarB";

	TTreeReaderValue<std::vector<B> > myVectorStarB (myTreeReader, finalBranchName);

	// Bool_t success = !myVectorStarB.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("vectorB dummies(%i):", myVectorStarB->size());

		for (int j = 0; j < LIST_ENTRIES; ++j){
			if (testValues && myVectorStarB->at(j).dummy != i * j * MULTIPLIER_VECTOR_STAR_B) success = false;
			if (printOut) printf(" %i", myVectorStarB->at(j).dummy);
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readVectorStarBArray(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "vectorStarB";

	TTreeReaderArray<B> myVectorStarB (myTreeReader, finalBranchName);

	// Bool_t success = !myVectorStarB.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("vectorB dummies(%i):", myVectorStarB.GetSize());

		for (int j = 0; j < LIST_ENTRIES; ++j){
			if (testValues && myVectorStarB.At(j).dummy != i * j * MULTIPLIER_VECTOR_STAR_B) success = false;
			if (printOut) printf(" %i", myVectorStarB.At(j).dummy);
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readVectorBArray(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "vectorB";

	TTreeReaderArray<B> myVectorB (myTreeReader, finalBranchName);

	// Bool_t success = !myVectorB.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("vectorB dummies(%i):", myVectorB.GetSize());

		for (int j = 0; j < LIST_ENTRIES && j < myVectorB.GetSize(); ++j){
			if (testValues && myVectorB.At(j).dummy != i * j * MULTIPLIER_VECTOR_B) success = false;
			if (printOut) printf(" %i", myVectorB.At(j).dummy);
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readBArray(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "BArray[12]";

	TTreeReaderArray<B> myBArray (myTreeReader, finalBranchName);

	// Bool_t success = !myBArray.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("BArray dummies(%i):", myBArray.GetSize());

		for (int j = 0; j < LIST_ENTRIES; ++j){
			if (testValues && myBArray.At(j).dummy != i * j * MULTIPLIER_B_ARRAY) success = false;
			if (printOut) printf(" %i", myBArray.At(j).dummy);
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readBStarArray(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "BStarArray";

	TTreeReaderArray<B> myBStarArray (myTreeReader, finalBranchName);

	// Bool_t success = !myBStarArray.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("BStarArray dummies(%i):", myBStarArray.GetSize());

		for (int j = 0; j < myBStarArray.GetSize(); ++j){
			if (testValues && myBStarArray.At(j).dummy != i * j * MULTIPLIER_B_STAR_ARRAY) success = false;
			if (printOut) printf(" %i", myBStarArray.At(j).dummy);
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readVectorBStarValue(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "vectorBStar";

	TTreeReaderValue<std::vector<B*> > myVectorBStar (myTreeReader, finalBranchName);

	// Bool_t success = !myVectorBStar.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("vectorBStar dummies(%i):", myVectorBStar->size());

		for (int j = 0; j < LIST_ENTRIES && j < myVectorBStar->size(); ++j){
			if (testValues && myVectorBStar->at(j)->dummy != i * j * MULTIPLIER_VECTOR_B_STAR) success = false;
			if (printOut) printf(" %i", myVectorBStar->at(j)->dummy);
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readVectorBStarArray(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "vectorBStar";

	TTreeReaderArray<B> myVectorBStar (myTreeReader, finalBranchName);

	// Bool_t success = !myVectorBStar.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("vectorBStar dummies(%i):", myVectorBStar.GetSize());

		for (int j = 0; j < LIST_ENTRIES && myVectorBStar.GetSize(); ++j){
			if (testValues && myVectorBStar.At(j).dummy != i * j * MULTIPLIER_VECTOR_B_STAR) success = false;
			if (printOut) printf(" %i", myVectorBStar.At(j).dummy);
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readBClonesArrayValue(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "BClonesArray";

	TTreeReaderValue<TClonesArray> myBClonesArray (myTreeReader, finalBranchName);

	// Bool_t success = !myBClonesArray.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("BClonesArray dummies(%i):", myBClonesArray->GetSize());

		for (int j = 0; j < LIST_ENTRIES && j < myBClonesArray->GetSize(); ++j){
			if (testValues && ((B*)myBClonesArray->At(j))->dummy != i * j * MULTIPLIER_B_CLONES_ARRAY) success = false;
			if (printOut) printf(" %i", ((B*)myBClonesArray->At(j))->dummy);
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readBClonesArrayArray(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "BClonesArray";

	TTreeReaderArray<B> myBClonesArray (myTreeReader, finalBranchName);

	// Bool_t success = !myBClonesArray.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("BClonesArray dummies(%i):", myBClonesArray.GetSize());

		for (int j = 0; j < LIST_ENTRIES && j < myBClonesArray.GetSize(); ++j){
			if (testValues && myBClonesArray.At(j).dummy != i * j * MULTIPLIER_B_CLONES_ARRAY) success = false;
			if (printOut) printf(" %i", myBClonesArray.At(j).dummy);
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readVectorBDummyArray(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "vectorB.dummy";

	TTreeReaderArray<Int_t> myVectorBDummyArray (myTreeReader, finalBranchName);

	// Bool_t success = !myVectorBDummyArray.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("vectorB.dummies(%i):", myVectorBDummyArray.GetSize());

		for (int j = 0; j < LIST_ENTRIES && j < myVectorBDummyArray.GetSize(); ++j){
			if (testValues && myVectorBDummyArray.At(j) != i * j * MULTIPLIER_VECTOR_B) success = false;
			if (printOut) printf(" %i", myVectorBDummyArray.At(j));
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

void readBClonesArrayDummyArray(const char* branchName = "A99.", Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = branchName;
	finalBranchName += "BClonesArray.dummy";

	TTreeReaderArray<Int_t> myBClonesArrayDummy (myTreeReader, finalBranchName);

	// Bool_t success = !myBClonesArrayDummy.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("BClonesArray.dummies(%i):", myBClonesArrayDummy.GetSize());

		for (int j = 0; j < LIST_ENTRIES && j < myBClonesArrayDummy.GetSize(); ++j){
			if (testValues && myBClonesArrayDummy.At(j) != i * j * MULTIPLIER_B_CLONES_ARRAY) success = false;
			if (printOut) printf(" %i", myBClonesArrayDummy.At(j));
		}

		if (printOut) printf("\n");
	}
	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}

// void readArrayOfBs(const char *branchName, const char *subBranchPostFix, Bool_t printOut = true, Bool_t testValues = false){
// 	TFile *myFile = TFile::Open("HardTreeFile.root");
// 	TTreeReader myTreeReader ("HardTree");

// 	Int_t multiplier;
// 	if (!strcmp(subBranchPostFix))

// 	TString finalBranchName = branchName;
// 	finalBranchName += subBranchPostFix;

// 	TTreeReaderArray<Int_t> myArrayOfBs (myTreeReader, finalBranchName);

// 	Bool_t successmyArrayOfBsRisValid();
// 	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
// 		if (printOut) printf("%s(%i):", subBranchPostFix, myArrayOfBs.GetSize());

// 		for (int j = 0; j < LIST_ENTRIES && i < myArrayOfBs.GetSize(); ++j){
// 			if (testValues && myArrayOfBs.At(j) != i * j * MULTIPLIER_B_CLONES_ARRAY) success = false;
// 			if (printOut) printf(" %i", myArrayOfBs.At(j));
// 		}

// 		if (printOut) printf("\n");
// 	}
// 	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
// }

// void readAObject(const char* branchName = "A99."){
// 	TFile *myFile = TFile::Open("HardTreeFile.root");
// 	TTreeReader myTreeReader ("HardTree");

// 	TTreeReaderValue<A> myAObject (myTreeReader, branchName);

// 	while (myTreeReademyAObjecNisValid();
// 		printf("BDummy: %i ", myAObject->BObject.dummy);
// 		printf("BStarDummy: %i\n", myAObject->BStar->dummy);
// 	}
// }

void readBranchVectorBValue(const char* splitLevel, Bool_t printOut = true, Bool_t testValues = false){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TString finalBranchName = "vectorB_";
	finalBranchName += splitLevel;

	TTreeReaderValue<std::vector<B> > myObject (myTreeReader, finalBranchName);

	// Bool_t success = !myObject.GetSetupStatus();
	Bool_t success = true;
	Bool_t read = false;
	for (int i = 1; myTreeReader.SetNextEntry(); ++i){
		read = true;
		if (printOut) printf("vectorB dummies:");

		for (int j = 0; j < myObject->size(); ++j){
			if (testValues && myObject->at(j).dummy != i * j * MULTIPLIER_VECTOR_B) success = false;
			if (printOut) printf(" %i", myObject->at(j).dummy);
		}

		if (printOut) printf("\n");
	}

	if (testValues) printf("%s\n", success && read ? "Success!" : "Failure");
}


void readTree(){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTree *myTree = (TTree*)myFile->Get("HardTree");
	myTree->Print();
	
	for (int i = 0; i < 10 && i < myTree->GetEntries(); ++i){
		myTree->Show(i);
	}

	myFile->Close();
}


void output(Bool_t printAll = false, Bool_t testAll = true){
	printf("A1: readNum(): ------------------------------ %s", printAll ? "\n": ""); readNum(					"A1.", printAll, testAll);
	printf("A1: readBObject(): -------------------------- %s", printAll ? "\n": ""); readBObject(				"A1.", printAll, testAll);
	printf("A1: readBStar(): ---------------------------- %s", printAll ? "\n": ""); readBStar(					"A1.", printAll, testAll);
	printf("A1: readVectorBValue(): --------------------- %s", printAll ? "\n": ""); readVectorBValue(			"A1.", printAll, testAll);
	printf("A1: readVectorStarBValue(): ----------------- %s", printAll ? "\n": ""); readVectorStarBValue(		"A1.", printAll, testAll);
	printf("A1: readVectorStarBArray(): ----------------- %s", printAll ? "\n": ""); readVectorStarBArray(		"A1.", printAll, testAll);
	printf("A1: readVectorBArray(): --------------------- %s", printAll ? "\n": ""); readVectorBArray(			"A1.", printAll, testAll);
	printf("A1: readBArray(): --------------------------- %s", printAll ? "\n": ""); readBArray(				"A1.", printAll, testAll);
	printf("A1: readBStarArray(): ----------------------- %s", printAll ? "\n": ""); readBStarArray(			"A1.", printAll, testAll);
	printf("A1: readVectorBStarValue(): ----------------- %s", printAll ? "\n": ""); readVectorBStarValue(		"A1.", printAll, testAll);
	printf("A1: readVectorBStarArray(): ----------------- %s", printAll ? "\n": ""); readVectorBStarArray(		"A1.", printAll, testAll);
	printf("A1: readBClonesArrayValue(): ---------------- %s", printAll ? "\n": ""); readBClonesArrayValue(		"A1.", printAll, testAll);
	printf("A1: readBClonesArrayArray(): ---------------- %s", printAll ? "\n": ""); readBClonesArrayArray(		"A1.", printAll, testAll);

	printf("A99: readNum(): ----------------------------- %s", printAll ? "\n": ""); readNum(					"A99.", printAll, testAll);
	printf("A99: readBObject(): ------------------------- %s", printAll ? "\n": ""); readBObject(				"A99.", printAll, testAll);
	printf("A99: readBObjectDummy(): -------------------- %s", printAll ? "\n": ""); readBObjectDummy(			"A99.", printAll, testAll);
	printf("A99: readBStar(): --------------------------- %s", printAll ? "\n": ""); readBStar(					"A99.", printAll, testAll);
	printf("A99: readVectorBValue(): -------------------- %s", printAll ? "\n": ""); readVectorBValue(			"A99.", printAll, testAll);
	printf("A99: readVectorStarBValue(): ---------------- %s", printAll ? "\n": ""); readVectorStarBValue(		"A99.", printAll, testAll);
	printf("A99: readVectorStarBArray(): ---------------- %s", printAll ? "\n": ""); readVectorStarBArray(		"A99.", printAll, testAll);
	printf("A99: readVectorBArray(): -------------------- %s", printAll ? "\n": ""); readVectorBArray(			"A99.", printAll, testAll);
	printf("A99: readBArray(): -------------------------- %s", printAll ? "\n": ""); readBArray(				"A99.", printAll, testAll);
	printf("A99: readBStarArray(): ---------------------- %s", printAll ? "\n": ""); readBStarArray(			"A99.", printAll, testAll);
	printf("A99: readVectorBStarValue(): ---------------- %s", printAll ? "\n": ""); readVectorBStarValue(		"A99.", printAll, testAll);
	printf("A99: readVectorBStarArray(): ---------------- %s", printAll ? "\n": ""); readVectorBStarArray(		"A99.", printAll, testAll);
	printf("A99: readBClonesArrayValue(): --------------- %s", printAll ? "\n": ""); readBClonesArrayValue(		"A99.", printAll, testAll);
	printf("A99: readBClonesArrayArray(): --------------- %s", printAll ? "\n": ""); readBClonesArrayArray(		"A99.", printAll, testAll);
	printf("A99: readVectorBDummyArray(): --------------- %s", printAll ? "\n": ""); readVectorBDummyArray(		"A99.", printAll, testAll);
	printf("A99: readBClonesArrayDummyArray(): ---------- %s", printAll ? "\n": ""); readBClonesArrayDummyArray("A99.", printAll, testAll);
	//printf("readAObject(): ------------------------ %s", printAll ? "\n": ""); readAObject();

	printf("A101: readNum(): ---------------------------- %s", printAll ? "\n": ""); readNum(					"A101.", printAll, testAll);
	printf("A101: readBObject(): ------------------------ %s", printAll ? "\n": ""); readBObject(				"A101.", printAll, testAll);
	printf("A101: readBStar(): -------------------------- %s", printAll ? "\n": ""); readBStar(					"A101.", printAll, testAll);
	printf("A101: readVectorBValue(): ------------------- %s", printAll ? "\n": ""); readVectorBValue(			"A101.", printAll, testAll);
	printf("A101: readVectorStarBValue(): --------------- %s", printAll ? "\n": ""); readVectorStarBValue(		"A101.", printAll, testAll);
	printf("A101: readVectorStarBArray(): --------------- %s", printAll ? "\n": ""); readVectorStarBArray(		"A101.", printAll, testAll);
	printf("A101: readVectorBArray(): ------------------- %s", printAll ? "\n": ""); readVectorBArray(			"A101.", printAll, testAll);
	printf("A101: readBArray(): ------------------------- %s", printAll ? "\n": ""); readBArray(				"A101.", printAll, testAll);
	printf("A101: readBStarArray(): --------------------- %s", printAll ? "\n": ""); readBStarArray(			"A101.", printAll, testAll);
	//printf("A101: readVectorBStarValue(): --------------- %s", printAll ? "\n": ""); readVectorBStarValue(			"A101.", printAll, testAll); // Segfault
	//printf("A101: readVectorBStarArray(): --------------- %s", printAll ? "\n": ""); readVectorBStarArray(			"A101.", printAll, testAll); // Segfault
	printf("A101: readBClonesArrayValue(): -------------- %s", printAll ? "\n": ""); readBClonesArrayValue(		"A101.", printAll, testAll);
	printf("A101: readBClonesArrayArray(): -------------- %s", printAll ? "\n": ""); readBClonesArrayArray(		"A101.", printAll, testAll);

	printf("S0_vectorB: readVectorBValue(): ------------- %s", printAll ? "\n": ""); readVectorBValue(			"S0_", 	printAll, testAll);
	// printf("vectorB_0: readBranchVectorBArray(): -------- %s", printAll ? "\n": ""); readBranchVectorBArray(		"S0_", 	printAll, testAll);

	printf("S99_: readNum(): ----------------------------- %s", printAll ? "\n": ""); readNum(						"S99_", printAll, testAll);
	printf("S99_: readBObject(): ------------------------- %s", printAll ? "\n": ""); readBObject(					"S99_", printAll, testAll);
	printf("S99_: readBObjectDummy(): -------------------- %s", printAll ? "\n": ""); readBObjectDummy(				"S99_", printAll, testAll);
	printf("S99_: readBStar(): --------------------------- %s", printAll ? "\n": ""); readBStar(					"S99_", printAll, testAll);
	printf("S99_: readVectorBValue(): -------------------- %s", printAll ? "\n": ""); readVectorBValue(				"S99_", printAll, testAll);
	printf("S99_: readVectorStarBValue(): ---------------- %s", printAll ? "\n": ""); readVectorStarBValue(			"S99_", printAll, testAll);
	printf("S99_: readVectorStarBArray(): ---------------- %s", printAll ? "\n": ""); readVectorStarBArray(			"S99_", printAll, testAll);
	printf("S99_: readVectorBArray(): -------------------- %s", printAll ? "\n": ""); readVectorBArray(				"S99_", printAll, testAll);
	//printf("S99_: readBArray(): -------------------------- %s", printAll ? "\n": ""); readBArray(					"S99_", printAll, testAll);
	printf("S99_: readBStarArray(): ---------------------- %s", printAll ? "\n": ""); readBStarArray(				"S99_", printAll, testAll);
	printf("S99_: readVectorBStarValue(): ---------------- %s", printAll ? "\n": ""); readVectorBStarValue(			"S99_", printAll, testAll);
	printf("S99_: readVectorBStarArray(): ---------------- %s", printAll ? "\n": ""); readVectorBStarArray(			"S99_", printAll, testAll);
	printf("S99_: readBClonesArrayValue(): --------------- %s", printAll ? "\n": ""); readBClonesArrayValue(		"S99_", printAll, testAll);
	printf("S99_: readBClonesArrayArray(): --------------- %s", printAll ? "\n": ""); readBClonesArrayArray(		"S99_", printAll, testAll);
	printf("S99_: readVectorBDummyArray(): --------------- %s", printAll ? "\n": ""); readVectorBDummyArray(		"S99_", printAll, testAll);
	printf("S99_: readBClonesArrayDummyArray(): ---------- %s", printAll ? "\n": ""); readBClonesArrayDummyArray(	"S99_", printAll, testAll);
}

void testAll(){
	output(false, true);
}

void printAll(){
	output(true, true);
}
