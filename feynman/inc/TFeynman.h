#include <iostream>
#include "TMath.h"
#include "TVector3.h"
#include <TLine.h>

class TFeynman {
	
	public:

		TFeynman(Double_t canvasWidth, Double_t canvasHeight){
			TCanvas *c1 = new TCanvas("c1", "c1", 10,10, canvasWidth, canvasHeight);
   			c1->Range(0, 0, 140, 60);
			gStyle->SetLineWidth(3);
		}

		// Fermions:
		// Quarks:
		void Quark(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, Double_t quarkName);

		// Leptons;
		void Lepton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, Double_t leptonName);
		void AntiLepton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, Double_t leptonName);

		// Bosons: 
		void StraightPhoton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY);
		void CurvedPhoton(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY);

		void StraightGluon(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY);
		void CurvedGluon(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY);
};
