/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// Plot the Amplitude of a Hydrogen Atom.
///
/// Visualize the Amplitude of a Hydrogen Atom in the n = 2, l = 0, m = 0 state.
/// Demonstrates how TH2F and TGraph2D can be used in Quantum Mechanics.
///
/// \macro_image
/// \macro_code
/// 
/// \author Advait Dhingra

#include <cmath>

double WaveFunction(double x, double y) {
    double r = sqrt(x *x + y*y);

    double w = (1/pow((4*sqrt(2*TMath::Pi())* 1), 1.5)) * (2 - (r / 1)*pow(TMath::E(), (-1 * r)/2)); // Wavefunction formula for psi 2,0,0

    return w*w; // amplitude

}

void schroedinger_hydrogen() {
   TH2F *h2D = new TH2F("Hydrogen Atom",
                        "#Psi^{2}_{200} i.e. n = 2, l = 0, m = 0; Position in x direction; Position in y direction",
                        1000, -10, 10, 1000, -10, 10); // for 2D view

   TGraph2D *h3D = new TGraph2D(); // for 3D view
   h3D->SetTitle("3D view of probability amplitude");

   for (float i = -10; i < 10; i += 0.01) {
      for (float j = -10; j < 10; j += 0.01) {
         h2D->Fill(i, j, WaveFunction(i, j));
         h3D->SetPoint(h3D->GetN(), i, j, WaveFunction(i, j));
      }
    }
    
    gStyle->SetPalette(53);
    gStyle->SetOptStat(0);
    TCanvas *c1 = new TCanvas("c1", "Schroedinger's Hydrogen Atom", 1500, 750);
    c1->Divide(2, 1);

    c1->cd(1);
    h2D->Draw("colz");

    c1->cd(2);
    h3D->Draw("surf1");
}

