// tutorial illustrating the use of the Student and F distributions
// author: Anna Kreshuk

#include "TMath.h"
#include "TF1.h"
#include "TCanvas.h"
#include <Riostream.h>
#include "TLegend.h"
#include "TLegendEntry.h"

void mathStudent()
{
  //drawing the set of student density functions
  //normal(0, 1) density drawn for comparison
  TCanvas *DistCanvas = new TCanvas("DistCanvas", "Distribution graphs", 10, 10, 1000, 800);
  DistCanvas->SetFillColor(17);
  DistCanvas->Divide(2, 2);
  DistCanvas->cd(1);
  gPad->SetGrid();
  gPad->SetFrameFillColor(19);
  TLegend *leg = new TLegend(0.6, 0.7, 0.89, 0.89);


  TF1* fgaus = new TF1("gaus", "TMath::Gaus(x, [0], [1], [2])", -5, 5);
  fgaus->SetTitle("Student density");
  fgaus->SetLineStyle(2);
  fgaus->SetLineWidth(1);
  fgaus->SetParameters(0, 1, kTRUE);
  leg->AddEntry(fgaus->DrawCopy(), "Normal(0,1)", "l");

  TF1* student = new TF1("student", "TMath::Student(x,[0])", -5, 5);
  //student->SetTitle("Student density");
  student->SetLineWidth(1);
  student->SetParameter(0, 10);
  student->SetLineColor(4);
  leg->AddEntry(student->DrawCopy("lsame"), "10 degrees of freedom", "l");

  student->SetParameter(0, 3);
  student->SetLineColor(2);
  leg->AddEntry(student->DrawCopy("lsame"), "3 degrees of freedom", "l");

  student->SetParameter(0, 1);
  student->SetLineColor(1);
  leg->AddEntry(student->DrawCopy("lsame"), "1 degree of freedom", "l");

  leg->Draw();

  //drawing the set of student cumulative probability functions
  DistCanvas->cd(2);
  gPad->SetFrameFillColor(19);
  gPad->SetGrid();
  TF1 *studentI = new TF1("studentI", "TMath::StudentI(x, [0])", -5, 5);
  studentI->SetTitle("Student cumulative dist.");
  studentI->SetLineWidth(1);
  TLegend *leg2 = new TLegend(0.6, 0.4, 0.89, 0.6);

  studentI->SetParameter(0, 10);
  studentI->SetLineColor(4);
  leg2->AddEntry(studentI->DrawCopy(), "10 degrees of freedom", "l");

  studentI->SetParameter(0, 3);
  studentI->SetLineColor(2);
  leg2->AddEntry(studentI->DrawCopy("lsame"), "3 degrees of freedom", "l");

  studentI->SetParameter(0, 1);
  studentI->SetLineColor(1);
  leg2->AddEntry(studentI->DrawCopy("lsame"), "1 degree of freedom", "l");
  leg2->Draw();

  //drawing the set of F-dist. densities
  TF1* fDist = new TF1("fDist", "TMath::FDist(x, [0], [1])", 0, 2);
  fDist->SetTitle("F-Dist. density");
  fDist->SetLineWidth(1);
  TLegend* legF1 = new TLegend(0.7, 0.7, 0.89, 0.89);

  DistCanvas->cd(3);
  gPad->SetFrameFillColor(19);
  gPad->SetGrid();

  fDist->SetParameters(1, 1);
  fDist->SetLineColor(1);
  legF1->AddEntry(fDist->DrawCopy(), "N=1 M=1", "l");

  fDist->SetParameter(1, 10);
  fDist->SetLineColor(2);
  legF1->AddEntry(fDist->DrawCopy("lsame"), "N=1 M=10", "l");

  fDist->SetParameters(10, 1);
  fDist->SetLineColor(8);
  legF1->AddEntry(fDist->DrawCopy("lsame"), "N=10 M=1", "l");

  fDist->SetParameters(10, 10);
  fDist->SetLineColor(4);
  legF1->AddEntry(fDist->DrawCopy("lsame"), "N=10 M=10", "l");

  legF1->Draw();

  //drawing the set of F cumulative dist.functions
  TF1* fDistI = new TF1("fDist", "TMath::FDistI(x, [0], [1])", 0, 2);
  fDistI->SetTitle("Cumulative dist. function for F");
  fDistI->SetLineWidth(1);
  TLegend* legF2 = new TLegend(0.7, 0.3, 0.89, 0.5);

  DistCanvas->cd(4);
  gPad->SetFrameFillColor(19);
  gPad->SetGrid();
  fDistI->SetParameters(1, 1);
  fDistI->SetLineColor(1);
  legF2->AddEntry(fDistI->DrawCopy(), "N=1 M=1", "l");

  fDistI->SetParameters(1, 10);
  fDistI->SetLineColor(2);
  legF2->AddEntry(fDistI->DrawCopy("lsame"), "N=1 M=10", "l");

  fDistI->SetParameters(10, 1);
  fDistI->SetLineColor(8);
  legF2->AddEntry(fDistI->DrawCopy("lsame"), "N=10 M=1", "l");

  fDistI->SetParameters(10, 10);
  fDistI->SetLineColor(4);
  legF2->AddEntry(fDistI->DrawCopy("lsame"), "N=10 M=10", "l");

  legF2->Draw();
  DistCanvas->cd();
}


