#include <iostream>
#include "src/TFeynman.cxx"


void test(){
   TFeynman *f = new TFeynman(800, 400);
   	f->Electron(10, 10, 30, 30, 6, 7);
	f->Electron(30, 30, 10, 50, 6, 45);
	f->StraightPhoton(30, 30, 70, 30, 50, 35);
	f->Electron(70, 30, 90, 50, 95, 50);
	f->Electron(90, 10, 70, 30, 85, 5);
	f->CurvedGluon(30, 30, 12, 135, 225, 22, 30);
}