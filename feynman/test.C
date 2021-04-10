#include <iostream>
#include "src/TFeynman.cxx"


void test(){
   TFeynman *f = new TFeynman(600, 300);

	f->Lepton(10, 10, 30, 30, 7, 6, "e", true);
	f->Lepton(10, 50, 30, 30, 5, 55, "e", false);
	f->CurvedPhoton(30, 30, 12.5*sqrt(2), 135, 225, 7, 30);
	f->Photon(30, 30, 55, 30, 42.5, 37.7);
	f->QuarkAntiQuark(70, 30, 15, 55, 45, "q");
	f->Gluon(70, 45, 70, 15, 77.5, 30);
	f->WeakBoson(85, 30, 110, 30, 100, 37.5, "Z^{0}");
	f->Quark(110, 30, 130, 50, 135, 55, "q", true);
	f->Quark(110, 30, 130, 10, 135, 6, "q", false);
	f->CurvedGluon(110, 30, 12.5*sqrt(2), 315, 45, 135, 30);
}
