/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This tutorial illustrates how you can conveniently apply BDTs in C++ using
/// the fast tree inference engine offered by TMVA. Supported workflows are
/// event-by-event inference, batch inference and pipelines with RDataFrame.
///
/// \macro_code
/// \macro_output
///
/// \date December 2018
/// \author Stefan Wunsch

using namespace TMVA::Experimental;

void tmva103_Application()
{
   // Load BDT model remotely from a webserver
   RBDT<> bdt("myBDT", "https://root.cern/files/tmva101.root");

   // Apply model on a single input
   auto y1 = bdt.Compute({1.0, 2.0, 3.0, 4.0});

   std::cout << "Apply model on a single input vector: " << y1[0] << std::endl;

   // Apply model on a batch of inputs
   float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
   RTensor<float> x(data, {2, 4});
   auto y2 = bdt.Compute(x);

   std::cout << "Apply model on an input tensor: " << y2 << std::endl;

   // Apply model as part of an RDataFrame workflow
   ROOT::RDataFrame df("Events", "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/SMHiggsToZZTo4L.root");
   auto df2 = df.Filter("nMuon >= 2")
                .Filter("nElectron >= 2")
                .Define("Muon_pt_1", "Muon_pt[0]")
                .Define("Muon_pt_2", "Muon_pt[1]")
                .Define("Electron_pt_1", "Electron_pt[0]")
                .Define("Electron_pt_2", "Electron_pt[1]")
                .Define("y",
                        Compute<4, float>(bdt),
                        {"Muon_pt_1", "Muon_pt_2", "Electron_pt_1", "Electron_pt_2"});

   std::cout << "Mean response on the signal sample: " << *df2.Mean("y") << std::endl;
}
