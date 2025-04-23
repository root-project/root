#include <TCanvas.h>
#include <TF1.h>
#include <TH1F.h>
#include <TPad.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TColor.h>
#include <TText.h>
#include <THStack.h>
#include <TPaveText.h>
#include <TGaxis.h>
#include <TApplication.h>
#include <vector>

#define MAX_RATIO_DIFF 2.0

void setPadStyle(TPad *pad) {
    pad->SetGrid(1, 1);
    pad->SetBorderMode(0);
    pad->SetFillStyle(0);
    pad->SetTopMargin(0.065);
    pad->SetRightMargin(0.045);
    pad->SetLeftMargin(0.125);
    pad->SetBottomMargin(0.11);
    pad->SetTickx();
    pad->SetTicky();
}

void gaussian_ratio_plot() {
    // Create a canvas
    TCanvas *c1 = new TCanvas("c1", "Canvas with Ratio Plot", 800, 800);

    // Create two pads: main plot and ratio plot
    TPad *pad1 = new TPad("pad1", "Main Plot", 0, 0.3, 0.75, 1.0);
    TPad *pad2 = new TPad("pad2", "Ratio Plot", 0, 0.05, 0.75, 0.3);
    TPad *pad3 = new TPad("pad3", "Projection", 0.75, 0.05, 1, 0.3);

    pad1->SetBottomMargin(0.02);
    pad2->SetTopMargin(0.02);
    pad2->SetBottomMargin(0.3);
    pad3->SetTopMargin(0.02);
    pad3->SetBottomMargin(0.3);
    pad3->SetLeftMargin(0.02);
    pad3->SetRightMargin(0.15);

    pad1->Draw();
    pad2->Draw();
    pad3->Draw();

    // Define two Gaussian functions
    TF1 *gauss1 = new TF1("gauss1", "gaus", -5, 5);
    gauss1->SetParameters(1, 0, 1); // amplitude, mean, sigma
    TF1 *gauss2 = new TF1("gauss2", "gaus", -5, 5);
    gauss2->SetParameters(1, 0, 2); // amplitude, mean, sigma

    // Create histograms and fill them with Gaussian distributions
    TH1F *h1 = new TH1F("h1", "Gaussian Distributions", 100, -5, 5);
    TH1F *h2 = new TH1F("h2", "Gaussian Distributions", 100, -5, 5);
    h1->FillRandom("gauss1", 10000);
    h2->FillRandom("gauss2", 10000);

    // Draw histograms in pad1
    pad1->cd();
    h1->SetLineColor(kRed);
    h1->Draw();
    h2->SetLineColor(kBlue);
    h2->Draw("SAME");

    // Create a legend and add histograms to it
    TLegend *legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(h1, "Gauss #1", "l");
    legend->AddEntry(h2, "Gauss #2", "l");
    legend->Draw();

    // Calculate ratio and plot in pad2
    TH1F *ratio = (TH1F*)h1->Clone("ratio");
    ratio->SetTitle("");
    ratio->Divide(h2);

    // Style the ratio plot
    ratio->GetYaxis()->SetTitle("Ratio");
    ratio->GetYaxis()->SetNdivisions(505);
    ratio->GetYaxis()->SetTitleSize(20);
    ratio->GetYaxis()->SetTitleFont(43);
    ratio->GetYaxis()->SetTitleOffset(1.55);
    ratio->GetYaxis()->SetLabelFont(43);
    ratio->GetYaxis()->SetLabelSize(15);
    ratio->GetXaxis()->SetTitleSize(20);
    ratio->GetXaxis()->SetTitleFont(43);
    ratio->GetXaxis()->SetTitleOffset(4.);
    ratio->GetXaxis()->SetLabelFont(43);
    ratio->GetXaxis()->SetLabelSize(15);
    ratio->SetMaximum(3.);
    ratio->SetMinimum(-3.);


    pad2->cd();
    ratio->Draw("ep");

    // Create and draw projection histogram in pad3
    pad3->cd();
    THStack *ratio_stack = new THStack("ratio_stack", "");
    TH1F *projection = new TH1F("projection", "Projection", 40, -3., 3.);
    for (int i = 1; i <= ratio->GetNbinsX(); ++i) {
        projection->Fill(ratio->GetBinContent(i));
    }
    projection->SetFillColorAlpha(kBlue, 0.6);
    projection->SetStats(false);
    ratio_stack->Add(projection);
    ratio_stack->Draw("hbar nostack");
}
