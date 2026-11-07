/// \file
/// \ingroup tutorial_ml
/// \notebook -nodraw
/// Macro evaluating a GNN model which was generated with the Parser macro
/// TMVA_SOFIE_GNN_Parser.py
///
/// \macro_code
///
/// \author

// need to add include path to find generated model file
#ifdef __CLING__
R__ADD_INCLUDE_PATH($PWD)
#endif

#include "encoder.hxx"
#include "core.hxx"
#include "decoder.hxx"
#include "output_transform.hxx"

#include "TRandom3.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TStopwatch.h"
#include "TMath.h"
#include "ROOT/RDataFrame.hxx"

#include <vector>

const int num_max_nodes = 100;
const int num_max_edges = 300;
const int NODE_FEATURE_SIZE = 4;
const int EDGE_FEATURE_SIZE = 4;
const int GLOBAL_FEATURE_SIZE = 1;
const int LATENT_SIZE = 100;

double check_mem(std::string s = ""){
   ProcInfo_t p;
   printf("%s - ",s.c_str());
   gSystem->GetProcInfo(&p);
   printf(" Rmem = %8.3f MB, Vmem = %8.f3 MB  \n",
          p.fMemResident /1024.,  /// convert memory from kB to MB
          p.fMemVirtual  /1024.
      );
   return p.fMemResident / 1024.;
}

// graph data for one event
struct GNN_Data {
   size_t num_nodes = 0;
   size_t num_edges = 0;
   std::vector<float> node_data;      // { num_nodes, node features }
   std::vector<float> edge_data;      // { num_edges, edge features }
   std::vector<float> global_data;    // { 1, global features }
   std::vector<int64_t> receivers;    // { num_edges }
   std::vector<int64_t> senders;      // { num_edges }
};

// concatenate the feature dimensions of two row-major {rows, f1} and {rows, f2} tensors
std::vector<float> ConcatenateFeatures(const std::vector<float> & a, const std::vector<float> & b, size_t rows)
{
   size_t fa = a.size() / rows, fb = b.size() / rows;
   std::vector<float> out(rows * (fa + fb));
   for (size_t i = 0; i < rows; i++) {
      std::copy(a.begin() + i * fa, a.begin() + (i + 1) * fa, out.begin() + i * (fa + fb));
      std::copy(b.begin() + i * fb, b.begin() + (i + 1) * fb, out.begin() + i * (fa + fb) + fa);
   }
   return out;
}

struct SOFIE_GNN {
   // the sessions are created for the maximum number of nodes/edges
   TMVA_SOFIE_encoder::Session encoder{"encoder.dat", num_max_edges, num_max_nodes};
   TMVA_SOFIE_core::Session core{"core.dat", num_max_edges, num_max_nodes};
   TMVA_SOFIE_decoder::Session decoder{"decoder.dat", num_max_edges, num_max_nodes};
   TMVA_SOFIE_output_transform::Session output_transform{"output_transform.dat", num_max_edges, num_max_nodes};

   // each session returns the {node, edge, global} output tensors
   std::vector<std::vector<float>> Infer(const GNN_Data & d, int nsteps) {
      auto latent = encoder.infer(d.num_nodes, d.node_data.data(), d.num_edges, d.edge_data.data(),
                                  d.global_data.data());
      auto latent0 = latent;
      std::vector<std::vector<float>> output;
      for (int i = 0; i < nsteps; i++) {
         auto node_input = ConcatenateFeatures(latent0[0], latent[0], d.num_nodes);
         auto edge_input = ConcatenateFeatures(latent0[1], latent[1], d.num_edges);
         auto global_input = ConcatenateFeatures(latent0[2], latent[2], 1);
         latent = core.infer(d.num_nodes, node_input.data(), d.num_edges, edge_input.data(),
                             global_input.data(), d.receivers.data(), d.senders.data());
         auto decoded = decoder.infer(d.num_nodes, latent[0].data(), d.num_edges, latent[1].data(),
                                      latent[2].data());
         output = output_transform.infer(d.num_nodes, decoded[0].data(), d.num_edges, decoded[1].data(),
                                         decoded[2].data());
      }
      return output;
   }
};

std::vector<GNN_Data> ReadData(std::string treename, std::string filename) {
   ROOT::RDataFrame df(treename,filename);
   auto ndata = df.Take<ROOT::RVec<float>>("node_data");
   auto edata = df.Take<ROOT::RVec<float>>("edge_data");
   auto gdata = df.Take<ROOT::RVec<float>>("global_data");
   auto rdata = df.Take<ROOT::RVec<int>>("receivers");
   auto sdata = df.Take<ROOT::RVec<int>>("senders");
   int nevts = ndata.GetPtr()->size();
   std::vector<GNN_Data> dataSet;
   dataSet.reserve(nevts);
   for (int i = 0; i < nevts; i++) {
      GNN_Data gd;
      auto & n = (*(ndata.GetPtr()))[i];
      auto & e = (*(edata.GetPtr()))[i];
      auto & g = (*(gdata.GetPtr()))[i];
      auto & r = (*(rdata.GetPtr()))[i];
      auto & s = (*(sdata.GetPtr()))[i];
      gd.num_nodes = n.size()/NODE_FEATURE_SIZE;
      gd.num_edges = e.size()/EDGE_FEATURE_SIZE;
      gd.node_data.assign(n.begin(), n.end());
      gd.edge_data.assign(e.begin(), e.end());
      gd.global_data.assign(g.begin(), g.end());
      gd.receivers.assign(r.begin(), r.end());
      gd.senders.assign(s.begin(), s.end());
      dataSet.emplace_back(std::move(gd));
   }
   return dataSet;
}


void TMVA_SOFIE_GNN_Application (bool verbose = false)
{
   check_mem("Initial memory");
   SOFIE_GNN gnn;
   check_mem("After creating GNN");

   const int nproc_steps = 5;

   std::cout << "reading data\n";
   auto inputData = ReadData("gdata","graph_data.root");
   int nevts = inputData.size();

   auto h1 = new TH1D("h1_sofie","SOFIE Node data",40,1,0);
   auto h2 = new TH1D("h2_sofie","SOFIE Edge data",40,1,0);
   auto h3 = new TH1D("h3_sofie","SOFIE Global data",40,1,0);
   std::cout << "doing inference...\n";

   check_mem("Before evaluating");
   TStopwatch w; w.Start();
   for (int i = 0; i < nevts; i++) {
      auto result = gnn.Infer(inputData[i], nproc_steps);
      // compute resulting means and plot them
      h1->Fill(TMath::Mean(result[0].begin(), result[0].end()));
      h2->Fill(TMath::Mean(result[1].begin(), result[1].end()));
      h3->Fill(TMath::Mean(result[2].begin(), result[2].end()));
   }
   w.Stop();
   w.Print();
   check_mem("End evaluation");
   auto c1 = new TCanvas("c1","SOFIE Results");
   c1->Divide(1,3);
   c1->cd(1); h1->Draw();
   c1->cd(2); h2->Draw();
   c1->cd(3); h3->Draw();

   // compare with the reference PyTorch result made by the Parser tutorial
   auto c2 = new TCanvas("c2","Reference Results");
   auto file = TFile::Open("graph_data.root");
   auto o1 = file->Get<TH1D>("h1");
   auto o2 = file->Get<TH1D>("h2");
   auto o3 = file->Get<TH1D>("h3");
   c2->Divide(1,3);
   c2->cd(1); o1->Draw();
   c2->cd(2); o2->Draw();
   c2->cd(3); o3->Draw();

   // check the mean of the global-data output distribution against the reference
   if (verbose)
      std::cout << "SOFIE global mean " << h3->GetMean() << "  reference " << o3->GetMean() << std::endl;
   if (std::abs(h3->GetMean() - o3->GetMean()) > 5e-4)
      std::cerr << "Error in comparing SOFIE and reference results" << std::endl;
}
