// @(#)root/feynman:$Id$
// Author: Advait Dhingra    12/04/2021

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>
#include "TMath.h"
#include "TVector3.h"
#include <TLine.h>

class TFeynman {
	
	public:

		// The constructor
		TFeynman(Double_t canvasWidth, Double_t canvasHeight){
			TCanvas *c1 = new TCanvas("c1", "c1", 10,10, canvasWidth, canvasHeight);
   			c1->Range(0, 0, 140, 60);
			gStyle->SetLineWidth(2);
		}

		// Fermions:
		// Quarks:
		void Quark(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, const char * quarkName, bool isMatter);
		// plots a Quark 
		void QuarkAntiQuark(Double_t x1, Double_t y1, Double_t rad, Double_t labelPositionX, Double_t labelPositionY, const char * quarkName);
		// plots a Quark-Antiquark pair in a cirlce
		// Leptons;
		void Lepton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, const char * whichLepton, bool isMatter);
		// plots a Lepton
		void LeptonAntiLepton(Double_t x1, Double_t y1, Double_t rad, Double_t labelPositionX, Double_t labelPositionY, const char * whichLepton, const char * whichAntiLepton);
		// plots a Lepton-Antilepton pair in a cirlce
		// Bosons: 
		void Photon(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY);
		// plots a Photon
		void CurvedPhoton(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY);
		// plots a Photon in a curved Arc

		void Gluon(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY);
		// plots a Gluon
		void CurvedGluon(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY);
		// plots a Gluon in a curved path

		void WeakBoson(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, const char *whichWeakBoson);
		// plots a Weak Boson
		void CurvedWeakBoson(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY, const char* whichWeakBoson);
		// plots a weak Boson in a curved path

		void Higgs(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY);
		// plots a Higgs Boson
		void CurvedHiggs(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY);
		// plots a Higgs Boson in a curved path
};
