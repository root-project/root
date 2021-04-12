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
        // plots a Quark
		void Quark(
            Double_t x1, // x-Coordinate of starting point
            Double_t y1, // y-coordinate of starting point
            Double_t x2, // x-coordinate of second point
            Double_t y2, // y-coordinate of second point
            Double_t labelPositionX, // x-coordinate of the particle-label
            Double_t labelPositionY, //y-coordinate of the particle label 
            const char * quarkName, // name of the quark (either u, d, c, s, t, b or q)
            bool isMatter // is the particle matter (if false then antimatter)
        );
		 // plots a Quark-Antiquark pair in a cirlce
		void QuarkAntiQuark(
            Double_t x1, // 
            Double_t y1, // coordinates of center of the circle
            Double_t rad, // radius of the circle
            Double_t labelPositionX, // 
            Double_t labelPositionY, // coordinates of the label (just define the coordinates of the first label)
            const char * quarkName // name of the first quark (u, d, c, s, t, b, q), antiparticle is automatically generated
            );
                
		// Leptons:
        // plots a Lepton
		void Lepton(Double_t x1, //
            Double_t y1, // starting coordinates
            Double_t x2, 
            Double_t y2, // stopping coordinates
            Double_t labelPositionX, //
            Double_t labelPositionY, // label coordinates
            const char * whichLepton, // name of the lepton (e, en, m, mn, t, tn) -> Electron, Electron-Neutrino etc.
            bool isMatter // true or false if it is matter or antimatter
            );
		// plots a Lepton-Antilepton pair in a cirlce
		void LeptonAntiLepton(Double_t x1, //
            Double_t y1, // coordinates of center of the circle
            Double_t rad, // radius of the circle
            Double_t labelPositionX, //
            Double_t labelPositionY, // coordinates of the circle
            const char * whichLepton, // name of the first lepton
            const char * whichAntiLepton // name of the antilepton);
        ); 
		
		// Bosons: 
        // Guage Bosons:
        // plots a Photon
		void Photon(Double_t x1, 
            Double_t y1, // starting coordinates
            Double_t x2, 
            Double_t y2, // stopping coordinates
            Double_t labelPositionX, 
            Double_t labelPositionY // label coordinates
        );
		// plots a Photon in a curved Arc
		void CurvedPhoton( 
            Double_t x1, 
            Double_t y1, // coordinates of center of arc
            Double_t rad, // radius of arc
            Double_t phimin, // minimum angle (see TArc)
            Double_t phimax, // maximum angle (See TAcr)
            Double_t labelPositionX, 
            Double_t labelPositionY // position of label
        );
		
        // plots a Gluon
		void Gluon(
            Double_t x1, 
            Double_t y1, // starting position
            Double_t x2, 
            Double_t y2, // stopping position
            Double_t labelPositionX, 
            Double_t labelPositionY // label position
        );
		// plots a Gluon in a curved path
		void CurvedGluon(
            Double_t x1, 
            Double_t y1, // position of center of Arc
            Double_t rad, // radius of arc
            Double_t phimin, // maximum angle (see TArc)
            Double_t phimax, //minimum angle (see TArc)
            Double_t labelPositionX, 
            Double_t labelPositionY // position of label
        );
		
        // plots a Weak Boson
		void WeakBoson( 
            Double_t x1, 
            Double_t y1, // starting position
            Double_t x2, 
            Double_t y2, // stopping position
            Double_t labelPositionX, 
            Double_t labelPositionY, // label position
            const char *whichWeakBoson // name of weak force boson in Latex (Z_{0}, W^{+}, W^{-})
        );
		// plots a weak Boson in a curved path
		void CurvedWeakBoson(
            Double_t x1, 
            Double_t y1, // position of center of arc
            Double_t rad, // radius of arc
            Double_t phimin, // maximum angle (See TArc)
            Double_t phimax, // minimum angle (see TArc)
            Double_t labelPositionX, 
            Double_t labelPositionY,// position of the label 
            const char* whichWeakBoson // name of the weak boson in Latex (Z_{0}, W^{+}, W^{-})
        );
		// Scalar Bosons:
        // plots a Higgs Boson
		void Higgs(
            Double_t x1, 
            Double_t y1, // starting position
            Double_t x2, 
            Double_t y2, // stopping position
            Double_t labelPositionX, 
            Double_t labelPositionY // label position
        );
		
        // plots a Higgs Boson in a curved path
		void CurvedHiggs(
            Double_t x1, 
            Double_t y1, // position of center of Arc
            Double_t rad, // radius of arc
            Double_t phimin, //maximum angle (see TArc)
            Double_t phimax, //minimum angle (see TArc)
            Double_t labelPositionX, 
            Double_t labelPositionY // position of label
        );
		
};
