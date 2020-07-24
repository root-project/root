/// \file
/// \ingroup tutorial_roostats
/// \notebook -js
/// \brief High Level Factory: creating a complex combined model.
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Danilo Piparo

#include <fstream>
#include "TString.h"
#include "TFile.h"
#include "TROOT.h"
#include "RooGlobalFunc.h"
#include "RooWorkspace.h"
#include "RooRealVar.h"
#include "RooAbsPdf.h"
#include "RooDataSet.h"
#include "RooPlot.h"
#include "RooStats/HLFactory.h"

// use this order for safety on library loading
using namespace RooFit;
using namespace RooStats;
using namespace std;

void rs603_HLFactoryElaborateExample()
{

   // --- Prepare the 2 needed datacards for this example ---

   TString card_name("rs603_card_WsMaker.rs");
   ofstream ofile(card_name);
   ofile << "// The simplest card for combination\n\n";
   ofile << "gauss1 = Gaussian(x[0,100],mean1[50,0,100],4);\n";
   ofile << "flat1 = Polynomial(x,0);\n";
   ofile << "sb_model1 = SUM(nsig1[120,0,300]*gauss1 , nbkg1[100,0,1000]*flat1);\n\n";
   ofile << "echo In the middle!;\n\n";
   ofile << "gauss2 = Gaussian(x,mean2[80,0,100],5);\n";
   ofile << "flat2 = Polynomial(x,0);\n";
   ofile << "sb_model2 = SUM(nsig2[90,0,400]*gauss2 , nbkg2[80,0,1000]*flat2);\n\n";
   ofile << "echo At the end!;\n";
   ofile.close();

   TString card_name2("rs603_card.rs");
   ofstream ofile2(card_name2);
   ofile2 << "// The simplest card for combination\n\n";
   ofile2 << "gauss1 = Gaussian(x[0,100],mean1[50,0,100],4);\n";
   ofile2 << "flat1 = Polynomial(x,0);\n";
   ofile2 << "sb_model1 = SUM(nsig1[120,0,300]*gauss1 , nbkg1[100,0,1000]*flat1);\n\n";
   ofile2 << "echo In the middle!;\n\n";
   ofile2 << "gauss2 = Gaussian(x,mean2[80,0,100],5);\n";
   ofile2 << "flat2 = Polynomial(x,0);\n";
   ofile2 << "sb_model2 = SUM(nsig2[90,0,400]*gauss2 , nbkg2[80,0,1000]*flat2);\n\n";
   ofile2 << "#include rs603_included_card.rs;\n\n";
   ofile2 << "echo At the end!;\n";
   ofile2.close();

   TString card_name3("rs603_included_card.rs");
   ofstream ofile3(card_name3);
   ofile3 << "echo Now reading the included file!;\n\n";
   ofile3 << "echo Including datasets in a Workspace in a Root file...;\n";
   ofile3 << "data1 = import(rs603_infile.root,\n";
   ofile3 << "               rs603_ws,\n";
   ofile3 << "               data1);\n\n";
   ofile3 << "data2 = import(rs603_infile.root,\n";
   ofile3 << "               rs603_ws,\n";
   ofile3 << "               data2);\n";
   ofile3.close();

   // --- Produce the two separate datasets into a WorkSpace ---

   HLFactory hlf("HLFactoryComplexExample", "rs603_card_WsMaker.rs", false);

   auto x = static_cast<RooRealVar *>(hlf.GetWs()->arg("x"));
   auto pdf1 = hlf.GetWs()->pdf("sb_model1");
   auto pdf2 = hlf.GetWs()->pdf("sb_model2");

   RooWorkspace w("rs603_ws");

   auto data1 = pdf1->generate(RooArgSet(*x), Extended());
   data1->SetName("data1");
   w.import(*data1);

   auto data2 = pdf2->generate(RooArgSet(*x), Extended());
   data2->SetName("data2");
   w.import(*data2);

   // --- Write the WorkSpace into a rootfile ---

   TFile outfile("rs603_infile.root", "RECREATE");
   w.Write();
   outfile.Close();

   cout << "-------------------------------------------------------------------\n"
        << " Rootfile and Workspace prepared \n"
        << "-------------------------------------------------------------------\n";

   HLFactory hlf_2("HLFactoryElaborateExample", "rs603_card.rs", false);

   x = hlf_2.GetWs()->var("x");
   pdf1 = hlf_2.GetWs()->pdf("sb_model1");
   pdf2 = hlf_2.GetWs()->pdf("sb_model2");

   hlf_2.AddChannel("model1", "sb_model1", "flat1", "data1");
   hlf_2.AddChannel("model2", "sb_model2", "flat2", "data2");

   auto data = hlf_2.GetTotDataSet();
   auto pdf = hlf_2.GetTotSigBkgPdf();
   auto thecat = hlf_2.GetTotCategory();

   // --- Perform extended ML fit of composite PDF to toy data ---
   pdf->fitTo(*data);

   // --- Plot toy data and composite PDF overlaid ---
   auto xframe = x->frame();

   data->plotOn(xframe);
   thecat->setIndex(0);
   pdf->plotOn(xframe, Slice(*thecat), ProjWData(*thecat, *data));

   thecat->setIndex(1);
   pdf->plotOn(xframe, Slice(*thecat), ProjWData(*thecat, *data));

   gROOT->SetStyle("Plain");

   xframe->Draw();
}
