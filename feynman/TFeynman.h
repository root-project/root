#include <iostream>
#include "TMath.h"
#include "TVector3.h"
#include <TLine.h>

class TFeynman {
	
	public:

		TFeynman(){
			cout << "Initialized TFeynman" << endl;
		}

		// Fermions:
		//Leptons;
		void Electron(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY);
		void Positron(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY);

		// Bosons: 
		void StraightPhoton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY);
		void CurvedPhoton(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY);

};
