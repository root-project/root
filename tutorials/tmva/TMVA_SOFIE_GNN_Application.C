// Macor evaluating a GNN model which was generated with the Parser macro
//

#include "encoder.hxx"
#include "core.hxx"
#include "decoder.hxx"
#include "output_transform.hxx"

#include "TMVA/SOFIE_common.hxx"
#include "TRandom3.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TTree.h"
#include "ROOT/RDataFrame.hxx"

using namespace TMVA::Experimental;
using namespace TMVA::Experimental::SOFIE;

void PrintTensor(RTensor<float> & t) {
   std::cout << " shape : " << ConvertShapeToString(t.GetShape()) << " size : " << t.GetSize() << " , ";
   std::cout << *t.begin();
   if (t.GetSize() > 2) std::cout << " ...";
   if (t.GetSize() > 1) std::cout << " " << *(t.end()-1);
   std::cout << std::endl;
}
void Print(GNN_Data & d, std::string txt = "") {
   if (!txt.empty()) std::cout << std::endl << txt << std::endl;
   std::cout << "node data:"; PrintTensor(d.node_data);
   std::cout << "edge data:"; PrintTensor(d.edge_data);
   std::cout << "global data:"; PrintTensor(d.global_data);
}

struct SOFIE_GNN {
   TMVA_SOFIE_encoder::Session encoder;
   TMVA_SOFIE_core::Session core;
   TMVA_SOFIE_decoder::Session decoder;
   TMVA_SOFIE_output_transform::Session output_transform;

   std::vector<GNN_Data> Infer(const GNN_Data & data, int nsteps) {
      // infer function
      auto input_data = Copy(data);
      encoder.infer(input_data);
      // latent0 is result of encoder. Need to copy because this stays the same
      auto latent0 = Copy(input_data);
      GNN_Data latent = input_data; // this can be a view
      std::vector<GNN_Data> outputData;
      for (int i = 0; i < nsteps; i++) {
         Print(latent0);
         Print(latent);
         auto core_input = Concatenate(latent0, latent,1);
         Print(core_input, "after concatenate");
         core.infer(core_input);
          Print(core_input, "after core inference");
         // here I need to copy
         latent = Copy(core_input);
         decoder.infer(core_input);
         output_transform.infer(core_input);
         outputData.push_back(Copy(core_input));
      }
      return outputData;
   }

};

const int num_max_nodes = 10;
const int num_max_edges = 30;
const int NODE_FEATURE_SIZE = 4;
const int EDGE_FEATURE_SIZE = 4;
const int GLOBAL_FEATURE_SIZE = 1;

std::vector<GNN_Data> GenerateData(int nevts, int seed) {
      TRandom3 r(seed);
      std::vector<GNN_Data> dataSet;
      dataSet.reserve(nevts);
      for (int i = 0; i < nevts; i++) {
         // generate first number of nodes and edges
        // size_t num_nodes = num_max_nodes;//r.Integer(num_max_nodes-2) + 2;
        // size_t num_edges = num_max_edges; //r.Integer(num_max_edges-1) + 1;
         size_t num_nodes = r.Integer(num_max_nodes-2) + 2;
         size_t num_edges = r.Integer(num_max_edges-1) + 1;
         GNN_Data gd;
         gd.node_data = RTensor<float>({num_nodes, NODE_FEATURE_SIZE});
         gd.edge_data = RTensor<float>({num_edges, EDGE_FEATURE_SIZE});
         gd.global_data = RTensor<float>({1, GLOBAL_FEATURE_SIZE});
         gd.receivers = std::vector<int>(num_edges);
         gd.senders = std::vector<int>(num_edges);
         auto genValue = [&]() { return r.Rndm()*10 -5; };
         auto genLink = [&] ()  { return r.Integer(num_nodes);};
         std::generate(gd.node_data.begin(), gd.node_data.end(), genValue);
         std::generate(gd.edge_data.begin(), gd.edge_data.end(), genValue);
         std::generate(gd.global_data.begin(), gd.global_data.end(), genValue);
         std::generate(gd.receivers.begin(), gd.receivers.end(), genLink);
         std::generate(gd.senders.begin(), gd.senders.end(), genLink);
         dataSet.emplace_back(gd);
      }
      return dataSet;
}

std::vector<GNN_Data> ReadData(std::string treename, std::string filename) {
   ROOT::RDataFrame df(treename,filename);
   auto ndata = df.Take<ROOT::RVec<float>>("node_data");
   auto edata = df.Take<ROOT::RVec<float>>("edge_data");
   auto gdata = df.Take<ROOT::RVec<float>>("global_data");
   auto rdata = df.Take<ROOT::RVec<int>>("receivers");
   auto sdata = df.Take<ROOT::RVec<int>>("senders");
   auto outdata = df.Take<ROOT::RVec<float>>("gnn_output");
   int nevts = ndata.GetPtr()->size();
   std::vector<GNN_Data> dataSet;
   dataSet.reserve(nevts);
   for (int i = 0; i < nevts; i++) {
      GNN_Data gd;
      auto & n = (*(ndata.GetPtr()))[i];
      size_t num_nodes = n.size()/NODE_FEATURE_SIZE;
      auto & e = (*(edata.GetPtr()))[i];
      size_t num_edges = e.size()/EDGE_FEATURE_SIZE;
      auto & g = (*(gdata.GetPtr()))[i];
      gd.node_data = RTensor<float>(n.data(), {num_nodes, NODE_FEATURE_SIZE});
      gd.edge_data = RTensor<float>(e.data(), {num_edges, EDGE_FEATURE_SIZE});
      gd.global_data =  RTensor<float>(g.data(), {1, GLOBAL_FEATURE_SIZE});
      auto & r = (*(rdata.GetPtr()))[i];
      auto & s = (*(sdata.GetPtr()))[i];
      gd.receivers.assign(r.begin(), r.end());
      gd.senders.assign(s.begin(), s.end());
      dataSet.emplace_back(Copy(gd)); // need to copy data in vector to own
   }
   return dataSet;
}

void PadData(std::vector<GNN_Data> & dataset) {
   for ( auto & gd : dataset) {
      gd.node_data.Resize({num_max_nodes, NODE_FEATURE_SIZE});
      gd.edge_data.Resize({num_max_edges, NODE_FEATURE_SIZE});
      // for the links I add all in the last node
      size_t nedges = gd.receivers.size();
      if (nedges < num_max_edges) {
         int npad = num_max_edges - nedges;
         gd.receivers.insert(gd.receivers.end(), npad, num_max_nodes-1);
         gd.senders.insert(gd.senders.end(), npad, num_max_nodes-1);
      }
   }
}

void TMVA_SOFIE_GNN_Application ()
{
   SOFIE_GNN gnn;
   const int seed = 111;
   const int nproc_steps = 5;
   // generate the input data
   int nevts = 100;

   //std::cout << "generating data\n";
   //auto inputData = GenerateData(nevts, seed);

   std::cout << "reading data\n";
   auto inputData = ReadData("gdata","graph_data.root");

   //std::cout << "padding data\n";
   //PadData(inputData) ;

   auto h1 = new TH1D("h1","Node data",40,1,0);
   auto h2 = new TH1D("h2","Edge data",40,1,0);
   auto h3 = new TH1D("h3","Global data",40,1,0);
   std::cout << "doing inference...\n";
   for (int i = 0; i < nevts; i++) {
      auto result = gnn.Infer(inputData[i], nproc_steps);
      // compute resulting mean and plot them
      auto & lr = result.back();
      h1->Fill(TMath::Mean(lr.node_data.begin(), lr.node_data.end()));
      h2->Fill(TMath::Mean(lr.edge_data.begin(), lr.edge_data.end()));
      h3->Fill(TMath::Mean(lr.global_data.begin(), lr.global_data.end()));
   }
   auto c1 = new TCanvas();
   c1->Divide(1,3);
   c1->cd(1); h1->Draw();
   c1->cd(2); h2->Draw();
   c1->cd(3); h3->Draw();


   // compare with the reference
   auto c2 = new TCanvas();
   auto file = TFile::Open("graph_data.root");
   auto o1 = file->Get("h1");
   auto o2 = file->Get("h2");
   auto o3 = file->Get("h3");
   c2->Divide(1,3);
   c2->cd(1); o1->Draw();
   c2->cd(2); o2->Draw();
   c2->cd(3); o3->Draw();

}

int main () {

   TMVA_SOFIE_GNN_Application();
}


