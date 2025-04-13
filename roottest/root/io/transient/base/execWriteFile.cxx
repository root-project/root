#ifndef __CLING__

#include "testobject.h"
#include "testobjectderived.h"

#include "TClass.h"
#include "TClonesArray.h"
#include "TFile.h"

#include "TInterpreter.h"

int WriteFile() {
	TestObj *bar = new TestObj;
	bar->store(42);
	printf("(main) Stored: %5.2lf\r\n", bar->retrieve());

	TestObjDerived *derivedbar = new TestObjDerived;
	derivedbar->storeInAnother(42);
	printf("(main) Stored: %5.2lf\r\n", derivedbar->retrieveFromAnother());
	TestObj *derivedbarasbar = derivedbar;
	derivedbarasbar->store(42);
	printf("(main) Stored: %5.2lf\r\n", derivedbarasbar->retrieve());

	TClonesArray *cloney1 = new TClonesArray("TestObjDerived", 10);

	for (Int_t i=0; i<10; i++) {
		new ((*cloney1)[i]) TestObjDerived(*derivedbar);
	}

	TClonesArray *cloney2 = (TClonesArray*)cloney1->Clone();

	TFile *dataFile1 = new TFile("data1.root", "RECREATE");
	dataFile1->Add(cloney1);
   dataFile1->Write();
	dataFile1->Close();
	//delete dataFile1;
	//delete cloney1;

	TFile *dataFile2 = new TFile("data2.root", "RECREATE");
	dataFile2->Add(cloney2);
   dataFile2->Write();
	dataFile2->Close();
	//delete dataFile2;
	//delete cloney2;

	return 0;
}
#else
int WriteFile();
#endif

int execWriteFile() {
   return WriteFile();
}