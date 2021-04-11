#include <iostream>
#include "TMath.h"
#include "TVector3.h"
#include <TLine.h>

class TFeynman {
	
	public:

		TFeynman(Double_t canvasWidth, Double_t canvasHeight){
			TCanvas *c1 = new TCanvas("c1", "c1", 10,10, canvasWidth, canvasHeight);
   			c1->Range(0, 0, 140, 60);
			gStyle->SetLineWidth(2);
		}

		// Fermions:
		// Quarks:
		void Quark(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, const char * quarkName, bool isMatter);
		void QuarkAntiQuark(Double_t x1, Double_t y1, Double_t rad, Double_t labelPositionX, Double_t labelPositionY, const char * quarkName);
		// Leptons;
		void Lepton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, const char * whichLepton, bool isMatter);
		void LeptonAntiLepton(Double_t x1, Double_t y1, Double_t rad, Double_t labelPositionX, Double_t labelPositionY, const char * whichLepton, const char * whichAntiLepton);
		// Bosons: 
		void Photon(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY);
		void CurvedPhoton(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY);

		void Gluon(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY);
		void CurvedGluon(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY);

		void WeakBoson(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, const char *whichWeakBoson);
		void CurvedWeakBoson(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY, const char* whichWeakBoson);

		void Higgs(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY);
		void CurvedHiggs(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY);
};
