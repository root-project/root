/// \file
/// \ingroup tutorial_math
/// \notebook
/// Tutorial illustrating how to create a plot comparing a Breit Wigner to a Relativistic Breit Wigner
///
/// can be run with:
///
/// ~~~{.cpp}
///  root[0] .x BreitWigner.C
/// ~~~
///
/// \macro_image
/// \macro_code
///
/// \author Jack Lindon

#include "TMath.h"

#include <limits>
#include <string>
#include "TAxis.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TStyle.h" //For gStyle to remove stat box.


void plotTwoTGraphs(Double_t x[], Double_t y1[], Double_t y2[], const Int_t nPoints
	  , Double_t lowerXLimit, Double_t upperXLimit
	  , Double_t lowerYLimit, Double_t upperYLimit
	  , std::string legend1, std::string legend2
	  , std::string plotTitle1, std::string plotTitle2, std::string plotTitle3
	  , std::string pdfTitle){

  ///////////////////////////////////////////////////////
  //Define variables for plot aesthetics and positioning
  Double_t legendXPos = 0.63;    Double_t legendYPos = 0.85;
  Double_t legendXWidth = 0.29;  Double_t legendYHeight = 0.1;
  Double_t plotTitleXPos = 0.23; Double_t plotTitleYPos = 0.25;
  Double_t fontSize = 0.04;
  Double_t lineWidth = 2;
  bool setLimitPlotLogScale = true;
  std::string xAxisTitle = "E [GeV]"; std::string yAxisTitle = "Events";
  Double_t xAxisTitleOffset = 1; Double_t yAxisTitleOffset = 1.3;
  gStyle->SetOptStat(0);

  ///////////////////////////////////////////////////////
  // Initialize TGraphs
  TGraph* gr1 = new TGraph(nPoints,x,y1);
  TGraph* gr2 = new TGraph(nPoints,x,y2);
  gr1->SetLineWidth(lineWidth);
  gr2->SetLineWidth(lineWidth);
  gr1->SetLineColor(kBlack);
  gr2->SetLineColor(kBlue);

  /////////////////////////////////////////////////////////
  // Initialize canvas
  TCanvas* c1 = new TCanvas("c1","transparent pad",200,10,600,600);
  c1->SetLogy(setLimitPlotLogScale);
  c1->SetTicks(1,1);
  c1->SetRightMargin(0.02);
  c1->SetTopMargin(0.02);


  ///////////////////////////////////////////////////////
  //Make just a basic invisible TGraph just for the axes
  const Double_t axis_x[2] = {lowerXLimit,upperXLimit};
  const Double_t axis_y[2] = {lowerYLimit,upperYLimit};      
  TGraph* grAxis = new TGraph(2, axis_x, axis_y);
  grAxis->SetTitle("");
  grAxis->GetYaxis()->SetTitle(yAxisTitle.c_str());
  grAxis->GetXaxis()->SetTitle(xAxisTitle.c_str());
  grAxis->GetXaxis()->SetRangeUser(lowerXLimit,upperXLimit);
  grAxis->GetYaxis()->SetRangeUser(lowerYLimit,upperYLimit);
  grAxis->GetXaxis()->SetLabelSize(fontSize);
  grAxis->GetYaxis()->SetLabelSize(fontSize);
  grAxis->GetXaxis()->SetTitleSize(fontSize);
  grAxis->GetYaxis()->SetTitleSize(fontSize);
  grAxis->GetXaxis()->SetTitleOffset(xAxisTitleOffset);
  grAxis->GetYaxis()->SetTitleOffset(yAxisTitleOffset);
  grAxis->SetLineWidth(0);//So invisible

  ///////////////////////////////////////////////////////////
  // Make legend and set aesthetics
  auto legend = new TLegend(legendXPos,legendYPos,legendXPos+legendXWidth,legendYPos+legendYHeight);
  legend->SetFillStyle(0);
  legend->SetBorderSize(0);
  legend->SetTextSize(fontSize);
  legend->AddEntry(gr1,legend1.c_str(),"L");
  legend->AddEntry(gr2,legend2.c_str(),"L");


  /////////////////////////////////////////////////////////////
  // Add plot title to plot. Make in three lines so not crowded.
  // Shift each line down by shiftY
  float shiftY{0.037};
  TLatex*   tex_Title = new TLatex(plotTitleXPos,plotTitleYPos-0*shiftY,plotTitle1.c_str());
  tex_Title->SetNDC();
  tex_Title->SetTextFont(42);
  tex_Title->SetTextSize(fontSize);
  tex_Title->SetLineWidth(lineWidth);
  TLatex*   tex_Title2 = new TLatex(plotTitleXPos,plotTitleYPos-1*shiftY,plotTitle2.c_str());
  tex_Title2->SetNDC();
  tex_Title2->SetTextFont(42);
  tex_Title2->SetTextSize(fontSize);
  tex_Title2->SetLineWidth(lineWidth);
  TLatex*   tex_Title3 = new TLatex(plotTitleXPos,plotTitleYPos-2*shiftY,plotTitle3.c_str());
  tex_Title3->SetNDC();
  tex_Title3->SetTextFont(42);
  tex_Title3->SetTextSize(fontSize);
  tex_Title3->SetLineWidth(lineWidth);


  /////////////////////////////////////
  // Draw everything
  grAxis->Draw("AL");
  gr1->Draw("L same");
  gr2->Draw("L same");
  legend->Draw();
  tex_Title->Draw();    
  tex_Title2->Draw();    
  tex_Title3->Draw();    
  c1->RedrawAxis(); //Be sure to redraw axis AFTER plotting TGraphs otherwise TGraphs will be on top of tick marks and axis borders.

  gPad->Print(pdfTitle.c_str());
  
}


void BreitWigner(){

  /////////////////////////////////////////////////////////
  // Define x axis limits and steps for each plotted point
  const Int_t nPoints = 1000;
  Double_t xMinimum = 0; Double_t xMaximum = 13000;
  Double_t xStepSize = (xMaximum-xMinimum)/nPoints;

  ///////////////////////////////////////////////////////
  // Define arrays of (x,y) points.
  Double_t x[nPoints];
  Double_t y_nonRelBW[nPoints], y_relBW[nPoints];

  
  //////////////////////////////////
  // Define Breit-Wigner parameters
  Double_t width = 1350;
  Double_t sigma = 269.7899;
  Double_t median = 9000;

  ///////////////////////////////////////////////////
  // Loop over x axis range, filling in (x,y) points,
  // and finding y minimums and maximums for axis limit.
  Double_t yMinimum = std::numeric_limits<Double_t>::max();
  Double_t yMaximum = TMath::BreitWignerRelativistic(median,median,width); //y maximum is at x=median (and non relativistic = relativistic at median so choice of function does not matter).
  for (Int_t i=0;i<nPoints;i++) {
    Double_t currentX = xMinimum+i*xStepSize;
    x[i] = currentX;
    y_nonRelBW[i] = TMath::BreitWigner(currentX,median,width);
    y_relBW[i] = TMath::BreitWignerRelativistic(currentX,median,width);

    if (y_nonRelBW[i]<yMinimum){yMinimum = y_nonRelBW[i];}
    if (y_relBW[i]<yMinimum){yMinimum = y_relBW[i];}
  }

  plotTwoTGraphs(x, y_nonRelBW, y_relBW, nPoints
       , xMinimum, xMaximum //xAxis limits
       , yMinimum/4, yMaximum*4 //yAxis limits, expand for aesthetics.
       ,"NonRel BW", "Rel BW" //Legend entries
       , "Comparing BW", "M = " + std::to_string(int(round(median))) + " GeV", "#Gamma = " + std::to_string(int(round(width))) + " GeV" //Plot Title entry (three lines)
       , "BW_M"+std::to_string(int(round(median)))+"_Gamma" + std::to_string(int(round(width))) +".pdf)" //PDF file title.
       );
}
