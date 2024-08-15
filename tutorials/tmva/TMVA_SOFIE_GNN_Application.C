// Macor evaluating a GNN model which was generated with the Parser macro
//

// need to add include path to find generated model file
#ifdef __CLING__
R__ADD_INCLUDE_PATH($PWD)
R__ADD_INCLUDE_PATH($ROOTSYS/runtutorials)
#endif

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
#include "TSystem.h"
#include "ROOT/RDataFrame.hxx"

using namespace TMVA::Experimental;
using namespace TMVA::Experimental::SOFIE;

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


template<class T>
void PrintTensor(RTensor<T> & t) {
   std::cout << " shape : " << ConvertShapeToString(t.GetShape()) << " size : " << t.GetSize() << "\n";
   auto & shape = t.GetShape();
   auto p = t.GetData();
   size_t nrows = (shape.size() > 1) ? shape[0] : 1;
   size_t ncols = (shape.size() > 1) ? t.GetStrides()[0] : shape[0];
   for (size_t i = 0; i < nrows; i++) {
      for (size_t j = 0; j < ncols; j++)  {
         if (j==ncols-1) {
            if (j>10) std::cout << "... ";
            std::cout << *p << std::endl;
         }
         else if (j<10)
            std::cout << *p << ", ";
         p++;
      }
   }
   std::cout << std::endl;
}
void Print(GNN_Data & d, std::string txt = "") {
   if (!txt.empty()) std::cout << std::endl << txt << std::endl;
   std::cout << "node data:"; PrintTensor(d.node_data);
   std::cout << "edge data:"; PrintTensor(d.edge_data);
   std::cout << "global data:"; PrintTensor(d.global_data);
   std::cout << "edge index:"; PrintTensor(d.edge_index);
}

struct SOFIE_GNN {
   bool verbose = false;
   TMVA_SOFIE_encoder::Session encoder;
   TMVA_SOFIE_core::Session core;
   TMVA_SOFIE_decoder::Session decoder;
   TMVA_SOFIE_output_transform::Session output_transform;

   std::vector<GNN_Data> Infer(const GNN_Data & data, int nsteps) {
      // infer function
      auto input_data = Copy(data);
      if (verbose) Print(input_data,"input_data");
      encoder.infer(input_data);
      // latent0 is result of encoder. Need to copy because this stays the same
      auto latent0 = Copy(input_data);
      GNN_Data latent = input_data; // this can be a view
      std::vector<GNN_Data> outputData;
      for (int i = 0; i < nsteps; i++) {
         if (verbose) Print(latent0,"input decoded data");
         if (verbose) Print(latent,"latent data");
         auto core_input = Concatenate(latent0, latent,1);
         if (verbose) Print(core_input, "after concatenate");
         core.infer(core_input);
         if (verbose) Print(core_input, "after core inference");
         // here I need to copy
         latent = Copy(core_input);
         decoder.infer(core_input);
         output_transform.infer(core_input);
         outputData.push_back(Copy(core_input));
      }
      return outputData;
   }

   SOFIE_GNN(bool v = false) : verbose(v) {}

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
         gd.edge_index =  RTensor<int>({2, num_edges});

         auto genValue = [&]() { return r.Rndm()*10 -5; };
         auto genLink = [&] ()  { return r.Integer(num_nodes);};
         std::generate(gd.node_data.begin(), gd.node_data.end(), genValue);
         std::generate(gd.edge_data.begin(), gd.edge_data.end(), genValue);
         std::generate(gd.global_data.begin(), gd.global_data.end(), genValue);
         std::generate(gd.edge_index.begin(), gd.edge_index.end(), genLink);
         dataSet.emplace_back(gd);
      }
      return dataSet;
}

std::vector<GNN_Data> ReadData(std::string treename, std::string filename) {
   bool verbose = false;
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
      size_t num_nodes = n.size()/NODE_FEATURE_SIZE;
      auto & e = (*(edata.GetPtr()))[i];
      size_t num_edges = e.size()/EDGE_FEATURE_SIZE;
      auto & g = (*(gdata.GetPtr()))[i];
      gd.node_data = RTensor<float>(n.data(), {num_nodes, NODE_FEATURE_SIZE});
      gd.edge_data = RTensor<float>(e.data(), {num_edges, EDGE_FEATURE_SIZE});
      gd.global_data =  RTensor<float>(g.data(), {1, GLOBAL_FEATURE_SIZE});
      gd.edge_index =  RTensor<int>({2, num_edges});
      auto & r = (*(rdata.GetPtr()))[i];
      auto & s = (*(sdata.GetPtr()))[i];
      // need to copy receivers/senders in edge_index tensor
      std::copy(r.begin(), r.end(), gd.edge_index.GetData());
      std::copy(s.begin(), s.end(), gd.edge_index.GetData()+num_edges);

      dataSet.emplace_back(Copy(gd)); // need to copy data in vector to own
      if (i < 1 && verbose) Print(dataSet[i],"Input for Event" + std::to_string(i));
   }
   return dataSet;
}


void TMVA_SOFIE_GNN_Application (bool verbose = false)
{
   check_mem("Initial memory");
   SOFIE_GNN gnn;
   check_mem("After creating GNN");


   const int seed = 111;
   const int nproc_steps = 5;
   // generate the input data

   int nevts;
   //std::cout << "generating data\n";
   //nevts = 100;
   //auto inputData = GenerateData(nevts, seed);

   std::cout << "reading data\n";
   auto inputData = ReadData("gdata","graph_data.root");
   nevts = inputData.size();

   //std::cout << "padding data\n";
   //PadData(inputData) ;

   auto h1 = new TH1D("h1","SOFIE Node data",40,1,0);
   auto h2 = new TH1D("h2","SOFIE Edge data",40,1,0);
   auto h3 = new TH1D("h3","SOFIE Global data",40,1,0);
   std::cout << "doing inference...\n";


   check_mem("Before evaluating");
   TStopwatch w; w.Start();
   for (int i = 0; i < nevts; i++) {
      auto result = gnn.Infer(inputData[i], nproc_steps);
      // compute resulting mean and plot them
      auto & lr = result.back();
      if (i < 1 && verbose) Print(lr,"Output for Event" + std::to_string(i));
      h1->Fill(TMath::Mean(lr.node_data.begin(), lr.node_data.end()));
      h2->Fill(TMath::Mean(lr.edge_data.begin(), lr.edge_data.end()));
      h3->Fill(TMath::Mean(lr.global_data.begin(), lr.global_data.end()));
   }
   w.Stop();
   w.Print();
   check_mem("End evaluation");
   auto c1 = new TCanvas("c1","SOFIE Results");
   c1->Divide(1,3);
   c1->cd(1); h1->Draw();
   c1->cd(2); h2->Draw();
   c1->cd(3); h3->Draw();


   // compare with the reference
   auto c2 = new TCanvas("c2","Reference Results");
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


