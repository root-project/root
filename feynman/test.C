#include <iostream>
#include "TFeynman.cxx"


void test(){
	
	TCanvas *c1 = new TCanvas("c1", "A canvas", 10,10, 600, 300);
   
	TFeynman *f = new TFeynman(c1);

	f->Electron(10, 10, 30, 30, 6, 7);
	f->Electron(30, 30, 10, 50, 6, 45);
	f->StraightPhoton(30, 30, 70, 30, 50, 35);
	f->Electron(70, 30, 90, 50, 85, 45);
	f->Electron(90, 10, 70, 30, 85, 5);
	f->CurvedPhoton(30, 30, 12, 135, 225, 22, 30);
}
