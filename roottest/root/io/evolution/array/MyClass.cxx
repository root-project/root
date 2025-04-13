#ifndef MYCLASS
#define MYCLASS 1
#endif

#include "MyClass.h"
#include <stdlib.h>
#include <Riostream.h>
#include <TObject.h>

ClassImp(MyClass)

#if MYCLASS == 1
// old version, classdef = 1
MyClass::MyClass():
	TObject()
{
	for (Int_t i = 0; i<5; i++){
		farray[i] = -1;
	}
}

MyClass::MyClass(Int_t* array):
TObject()
{
	for (Int_t i = 0; i<5; i++){
		farray[i] = array[i];
	}
}

MyClass::~MyClass(){

}

void MyClass::SetArray(Int_t* array){

	//cout << " hello!" << endl;
	for (Int_t i = 0; i<5; i++){
		farray[i] = array[i];
		printf("farray = %d\n",farray[i]);
	}
	return;

}

#elif MYCLASS == 2
// new version, classdef = 2
MyClass::MyClass():
	TObject(),
	fentries(5),
	farrayPointer(0x0)
{
	farrayPointer = new Int_t[fentries];
	for (Int_t i = 0; i<5; i++){
		farrayPointer[i] = -1;
	}
}

MyClass::MyClass(Int_t* array):
	TObject(),
	fentries(5),
	farrayPointer(0x0)
{
	farrayPointer = new Int_t[fentries];
	for (Int_t i = 0; i<5; i++){
		farrayPointer[i] = array[i];
	}
}

MyClass::~MyClass(){

	if (farrayPointer) {
		delete [] farrayPointer;
	}
        farrayPointer = 0;
}

void MyClass::SetArray(Int_t* array){

	for (Int_t i = 0; i<5; i++){
		farrayPointer[i] = array[i];
		printf("farrayPointer = %d\n",farrayPointer[i]);
	}
	return;

}
#endif

#ifdef __MAKECINT__
#pragma link C++ class MyClass+;
#endif
