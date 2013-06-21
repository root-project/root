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
#pragma link C++ class TTreeReaderArray<double>+;
#pragma link C++ class TTreeReaderArray<B>+;
#endif

#define TREE_ENTRIES 10
#define LIST_ENTRIES 10

void makeTree(){
	TFile *myFile = new TFile("HardTreeFile.root", "RECREATE");
	TTree *myTree = new TTree("HardTree", "This is hard to read");

	A myObject0 (14);
	A myObject1 (14);
	A myObject99 (14);
	A myObject101 (14);

	myTree->Branch("A0.",	"A", 	&myObject0,		32000, 0);
	myTree->Branch("A1.",	"A", 	&myObject1,		32000, 1);
	myTree->Branch("A99.",	"A", 	&myObject99,	32000, 99);
	myTree->Branch("A101.",	"A", 	&myObject101,	32000, 101);

	for (int i = 1; i < TREE_ENTRIES + 1; ++i){
		printf("\nEntry %i\n\n", i);

		// Clear old values
		myObject0.ResetVectorB();
		myObject1.ResetVectorB();
		myObject99.ResetVectorB();
		myObject101.ResetVectorB();

		// Integer
		// printf("Setting nums\n");
		// myObject0.num = i;
		// myObject1.num = i*2;
		// myObject99.num = i*3;
		// myObject101.num = i*4;

		// myObject0.num = i;
		// myObject1.num = i*2;
		// myObject99.num = i*3;
		// myObject101.num = i*4;

		printf("Setting BObject\n");
		myObject0.BObject.dummy = i;
		myObject1.BObject.dummy = i*2;
		myObject99.BObject.dummy = i*3;
		myObject101.BObject.dummy = i*4;

		// std::vector<Int_t> *list1 = new std::vector<Int_t>();
		// list1->push_back(2);
		// printf("ListSize: %lu\n", list1->size());

		// std::vector<B> *myList = new std::vector<B>();

		// printf("Size: %lu\n", myList->size());

		for (int j = 0; j < LIST_ENTRIES; ++j){
			// Vector of objects
			B obj (i*j);
			printf("Adding %i to vectorB\n", i*j);
			myObject0.AddToVectorB(obj); 	obj.dummy++;
			myObject1.AddToVectorB(obj); 	obj.dummy++;
			myObject99.AddToVectorB(obj); 	obj.dummy++;
			myObject101.AddToVectorB(obj); 	obj.dummy++;

			// Hangs makeTree
			printf("Adding %i to vectorStarB\n", obj.dummy);
			myObject0.AddToVectorStarB(obj); 	obj.dummy++;
			myObject1.AddToVectorStarB(obj); 	obj.dummy++;
			myObject99.AddToVectorStarB(obj); 	obj.dummy++;
			myObject101.AddToVectorStarB(obj);

			// myList->push_back(obj);

			// Vector of pointers
			printf("Adding %i to vectorBStar\n", i*j*2);
			B *ptr = new B(i*j*2);
			myObject0.AddToVectorBStar(ptr);
			myObject1.AddToVectorBStar(ptr);
			myObject99.AddToVectorBStar(ptr);
			//myObject101.AddToVectorBStar(ptr); // Breaks readTree (segfault)
		}

		// myObject0.SetVectorStarB(myList);
		// myObject1.SetVectorStarB(myList);
		// myObject99.SetVectorStarB(myList);
		// myObject101.SetVectorStarB(myList);

		printf("Setting BStar\n");
		B *objPtr = new B(i);
		myObject0.SetBStar(objPtr);
		myObject1.SetBStar(objPtr);
		myObject99.SetBStar(objPtr);
		myObject101.SetBStar(objPtr);

		printf("Filling BArray\n");
		// B myArray[] = {B(12), B(11), B(10), B(9), B(8), B(7), B(6), B(5), B(4), B(3), B(2), B(1)}; // Breaks makeTree (segFault @ end)
		B *myArray = new B[12];
		for (int j = 0; j < 12; ++j){
			myArray[j].dummy = j * 2;
		}
		myObject0.FillBArray(myArray);
		myObject1.FillBArray(myArray);
		myObject99.FillBArray(myArray);
		myObject101.FillBArray(myArray);
		delete [] myArray;

		printf("Filling BStarArray\n");
		for (int j = 0; j < 14; ++j){
			myObject0.BStarArray[j] = i*j;
			myObject1.BStarArray[j] = i*j*2;
			myObject99.BStarArray[j] = i*j*3;
			myObject101.BStarArray[j] = i*j*4;
		}

		printf("Filling BClonesArray\n");
		for (int j = 0; j < LIST_ENTRIES; ++j ){
			((B*)myObject0.BClonesArray.New(j))->dummy = i*j;
			((B*)myObject1.BClonesArray.New(j))->dummy = i*j*2;
			((B*)myObject99.BClonesArray.New(j))->dummy = i*j*3;
			((B*)myObject101.BClonesArray.New(j))->dummy = i*j*3;
		}

		//printf("vectorB size: %i\n", myObject0.vectorB.size());
		myTree->Fill();

		// myObject0.ResetBStarArray();
		// myObject1.ResetBStarArray();
		// myObject99.ResetBStarArray();
		// myObject101.ResetBStarArray();

		// Hangs makeTree
		// myObject0.vectorStarB = new std::vector<B>();
		// myObject1.vectorStarB = new std::vector<B>();
		// myObject99.vectorStarB = new std::vector<B>();
		// myObject101.vectorStarB = new std::vector<B>();

		myObject0.ResetVectorStarB();
		myObject1.ResetVectorStarB();
		myObject99.ResetVectorStarB();
		myObject101.ResetVectorStarB();

		myObject0.vectorBStar.clear();
		myObject1.vectorBStar.clear();
		myObject99.vectorBStar.clear();
		myObject101.vectorBStar.clear();

		printf("Entry created\n");
	}
	printf("Tree created\n");

	myFile->Write();
}

void readNum(){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TTreeReaderValue<Int_t> myNum (myTreeReader, "A99.num");

	while (myTreeReader.SetNextEntry()){
		printf("Num: %i\n", *myNum);
	}
}

void readBObject(){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TTreeReaderValue<B> myBObject (myTreeReader, "A99.BObject");

	while (myTreeReader.SetNextEntry()){
		printf("Dummy: %i\n", myBObject->dummy);
	}
}

void readBObjectDummy(){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TTreeReaderValue<Int_t> myDummy (myTreeReader, "A99.BObject.dummy");

	while (myTreeReader.SetNextEntry()){
		printf("Dummy: %i\n", *myDummy);
	}
}

void readVectorBValue(){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TTreeReaderValue<std::vector<B> > myVectorB (myTreeReader, "A99.vectorB");

	while (myTreeReader.SetNextEntry()){
		printf("vectorB dummies:");

		for (int i = 0; i < LIST_ENTRIES; ++i){
			printf(" %i", myVectorB->at(i).dummy);
		}

		printf("\n");
	}
}

void readVectorBArray(){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TTreeReaderArray<B> myVectorB (myTreeReader, "A99.vectorB");

	while (myTreeReader.SetNextEntry()){
		printf("vectorB dummies(%i):", myVectorB.GetSize());

		for (int i = 0; i < LIST_ENTRIES && i < myVectorB.GetSize(); ++i){
			printf(" %i", myVectorB.At(i).dummy);
		}

		printf("\n");
	}
}

void readAObject(){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTreeReader myTreeReader ("HardTree");

	TTreeReaderValue<A> myAObject (myTreeReader, "A99.");

	while (myTreeReader.SetNextEntry()){
		printf("Num: %i\n", myAObject->num);
		printf("BDummy: %i\n", myAObject->BObject.dummy);
		printf("BStarDummy: %i\n", myAObject->BStar->dummy);
	}
}

void readTree(){
	TFile *myFile = TFile::Open("HardTreeFile.root");
	TTree *myTree = (TTree*)myFile->Get("HardTree");
	
	for (int i = 0; i < 10 && i < myTree->GetEntries(); ++i){
		myTree->Show(i);
	}

	myFile->Close();
}