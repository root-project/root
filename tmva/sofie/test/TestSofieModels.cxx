//#include "TPython.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TInterpreter.h"
//#include "TMacro.h"
#include <vector>
#include <fstream>
#include <limits>

#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

#include  "gtest/gtest.h"

bool verbose = true;
int sessionId = 0;

void ExecuteSofieParser(std::string modelName) {
   using namespace TMVA::Experimental;
   SOFIE::RModelParser_ONNX parser;
   std::string inputName = modelName + ".onnx";
   std::cout << "parsing file " << inputName << std::endl;
   SOFIE::RModel model = parser.Parse(inputName);
   std::cout << "generating model.....\n";
   model.Generate(1, 1);
   std::string outputName = modelName + ".hxx";
   std::cout << "writing model as header .....\n";
   model.OutputGenerated(); // outputName);
   std::cout << "output written in  " << outputName << std::endl;
}


int DeclareCode(std::string modelName)
{
   // increment session Id to avoid clash in session variable name
   sessionId++;
   // inference code for gInterpreter->Declare + gROOT->ProcessLine
   // one could also use TMacro build with correct signature
   // TMacro m("testSofie"); m.AddLine("std::vector<float> testSofie(float *x) { return s.infer(x);}")
   // std::vector<float> * result = (std::vector<float> *)m.Exec(Form(float*)0x%lx , xinput.data));
   std::string code = std::string("#include \"") + modelName + ".hxx\"\n";
   code += "TMVA_SOFIE_" + modelName + "::Session s" + std::to_string(sessionId) + ";\n";

   gInterpreter->Declare(code.c_str());
   return sessionId;  
}

std::vector<float> RunInference(float * x, int sId) {
   // run inference code using gROOT->ProcessLine
   printf("doing inference.....");
   TString cmd = TString::Format("s%d.infer( (float*)0x%lx )", sId,(ULong_t)x);
   if (!verbose)  cmd += ";";
   std::vector<float> *result = (std::vector<float> *)gROOT->ProcessLine(cmd);
   return *result;
}

void TestLinear(int nbatches, bool useBN = false, int inputSize = 10, int nlayers = 4)
{
   std::string modelName = "LinearModel";
   if (useBN) modelName += "_BN";
   modelName += "_B" + std::to_string(nbatches);

   // network parameters : nbatches, inputDim, nlayers
   std::vector<int> params = {nbatches, inputSize, nlayers};

   std::string command = "python3 LinearModelGenerator.py ";
   for (size_t i = 0; i < params.size(); i++)
      command += "  " + std::to_string(params[i]);
   if (useBN)
      command += "  --bn";

   printf("executing %s\n", command.c_str());
   gSystem->Exec(command.c_str());

   ExecuteSofieParser(modelName);

   int id = DeclareCode(modelName);

   // input data
   std::vector<float> xinput(nbatches * inputSize);
   for (int ib = 0; ib < nbatches; ib++) {
      std::vector<float> x1(inputSize, float(ib + 1));
      std::copy(x1.begin(), x1.end(), xinput.begin() + ib * inputSize);
   }

   auto result = RunInference(xinput.data(), id);

   // read reference value from test file
   std::vector<float> refValue(result.size());

   std::ifstream f(std::string(modelName + ".out").c_str());
   for (size_t i = 0; i < refValue.size(); ++i) {
      f >> refValue[i];
      if (verbose)
         std::cout << " result " << result.at(i) << " reference " << refValue[i] << std::endl;
      if (std::abs(refValue[i]) > 0.5)
         EXPECT_FLOAT_EQ(result.at(i), refValue[i]);
      else
         // expect float fails for small values
         EXPECT_NEAR(result.at(i), refValue[i], 10 * std::numeric_limits<float>::epsilon());
   }
}

void TestConv( std::string type, int nbatches, bool useBN = false, int ngroups = 2, int nchannels = 2, int nd = 4, int nlayers = 4, int usePool = 0)
{
   std::string modelName = "Conv" + type + "Model";
   if (useBN) modelName += "_BN";
   if (usePool == 1) modelName += "_MAXP";
   if (usePool == 2) modelName += "_AVGP";
   modelName += "_B" + std::to_string(nbatches);

   // input size is fixed to (nb, nc, nd, nd)
   int inputDim = nd;
   if (type == "2d") inputDim *= nd;
   if (type == "3d") inputDim *= nd*nd;

   const int inputSize = nchannels * inputDim;

   //const char *argv[5] = {}
   std::string argv[5];

   argv[0] = std::to_string(nbatches);
   //.c_str();
   argv[1] = std::to_string(nchannels);
   argv[2] = std::to_string(nd);
   argv[3] = std::to_string(ngroups); // for 3d this is depth size
   argv[4] = std::to_string(nlayers);
   std::string command = "python3 Conv" + type + "ModelGenerator.py ";
   for (int i = 0; i < 5; i++) {
      command += " ";
      command += argv[i];
   }
   if (useBN) command += "  --bn";
   if (usePool == 1) command += " --maxpool";
   if (usePool == 2) command += " --avgpool";
   printf("executing %s\n", command.c_str());
   gSystem->Exec(command.c_str());

   // some model needs some semplifications
   if (usePool == 2) {
      printf("simplify onnx model using onnxsim tool \n");
      std::string cmd = "python3 -m onnxsim " + modelName + ".onnx " + modelName + ".onnx";
      int ret = gSystem->Exec(cmd.c_str());
      if (ret != 0) {
         std::cout << "Error when simplifing ONNX model with AveragePool layer using onnx-simplifier (onnxsim) - skip the test" << std::endl;
         GTEST_SKIP();
         return;
      }
   }

  
   ExecuteSofieParser(modelName);

   int id = DeclareCode(modelName);

   // input data 
   std::vector<float> xinput(nbatches*inputSize);
   for (int ib = 0; ib < nbatches; ib++) {
      std::vector<float> x1(inputDim, float(ib + 1));
      std::vector<float> x2(inputDim, -float(ib + 1));
      // x1 and x2 are the two channels, if more channels will be with zero
      std::copy(x1.begin(), x1.end(), xinput.begin() + ib * inputSize);
      if (nchannels > 1)
         std::copy(x2.begin(), x2.end(), xinput.begin() + ib * inputSize + x1.size());
   }

   auto result = RunInference(xinput.data(), id);

   
   // read reference value from test file
   std::vector<float> refValue(result.size());

   std::ifstream f(std::string(modelName + ".out").c_str());
   for (size_t i = 0; i < refValue.size(); ++i) {
      f >> refValue[i];
      if (verbose) std::cout << " result " << result.at(i) << " reference " << refValue[i] << std::endl;
      if (std::abs(refValue[i]) > 0.5)
         EXPECT_FLOAT_EQ(result.at(i), refValue[i]);
      else 
         // expect float fails for small values
         EXPECT_NEAR(result.at(i), refValue[i], 10 * std::numeric_limits<float>::epsilon());
   }
}

void TestRecurrent(std::string type, int nbatches, int inputSize = 5, int seqSize = 10, int hiddenSize = 3, int nlayers = 1)
{

   if (type.empty()) type = "RNN";
   std::string modelName = type + "Model";
   modelName += "_B" + std::to_string(nbatches);

   // network parameters : nbatches, inputDim, nlayers
   std::vector<int> params = {nbatches, inputSize, seqSize, hiddenSize, nlayers};

   std::string command = "python3 RecurrentModelGenerator.py ";
   for (size_t i = 0; i < params.size(); i++)
      command += "  " + std::to_string(params[i]);
   if (type == "LSTM")
      command += "  --lstm";
   else if (type == "GRU")
      command += "  --gru";

   printf("executing %s\n", command.c_str());
   gSystem->Exec(command.c_str());
   // need to simplify obtained recurrent ONNX model
   printf("simplify onnx model using onnxsim tool \n");
   std::string cmd = "python3 -m onnxsim " + modelName + ".onnx " + modelName + ".onnx";
   int ret = gSystem->Exec(cmd.c_str());
   if (ret != 0) {
      std::cout << "Error when simplifing ONNX Recurrent model using onnx-simplifier (onnxsim) - skip the test" << std::endl;
      GTEST_SKIP();
      return;
   }

   ExecuteSofieParser(modelName);

   int id = DeclareCode(modelName);

   // input data
   std::vector<float> xinput(nbatches * seqSize * inputSize);
   for (int ib = 0; ib < nbatches; ib++) {
      for (int it = 0; it < seqSize; it++) {
         std::vector<float> x1(inputSize, std::pow(-1, ib + 2) * float(it + 1));
         std::copy(x1.begin(), x1.end(), xinput.begin() + ib * inputSize*seqSize + it * inputSize);
      }
   }
   if (verbose) {
      std::cout << " input data \n";
      int k = 0;
      for (int i = 0; i < nbatches; ++i) {
         for (int j = 0; j < seqSize; j++) {
            for (int l = 0; l < inputSize; l++) {
               std::cout << xinput[k++] << ", ";
            }
             std::cout << "\n";
         }
         std::cout << "\n";
      }
   }

   auto result = RunInference(xinput.data(), id);

   // read reference value from test file
   std::vector<float> refValue(result.size());

   std::ifstream f(std::string(modelName + ".out").c_str());
   for (size_t i = 0; i < refValue.size(); ++i) {
      f >> refValue[i];
      if (verbose)
         std::cout << " result " << result.at(i) << " reference " << refValue[i] << std::endl;
      if (std::abs(refValue[i]) > 0.5)
         EXPECT_FLOAT_EQ(result.at(i), refValue[i]);
      else
         // expect float fails for small values
         EXPECT_NEAR(result.at(i), refValue[i], 10 * std::numeric_limits<float>::epsilon());
   }
}

TEST(SOFIE, Linear_B1) {
   TestLinear(1);
}
TEST(SOFIE, Linear_B4)
{
   // test batch =4 (equal output size)
   TestLinear(4);
}
TEST(SOFIE,Conv2d_B1) {
   TestConv("2d", 1);
}
TEST(SOFIE, Conv2d_B4)
{
   TestConv("2d",4);
}
// test with batch normalization
TEST(SOFIE, Linear_BNORM_B8)
{
   TestLinear(8,true,5,4);
}
TEST(SOFIE, Conv2d_BNORM_B5)
{
   TestConv("2d",5,true);
}
// test with max pooling
TEST(SOFIE, Conv2d_MAXPOOL_B2)
{
   TestConv("2d",2,false,1,2,3,1,1);
}
// test with avg pooling
TEST(SOFIE, Conv2d_AVGPOOL_B2)
{
   TestConv("2d", 2, false, 1, 2, 4, 1, 2);
}

// test conv1d
TEST(SOFIE, Conv1d_B1)
{
   TestConv("1d", 1, false, 1, 2, 10, 1, 0);
}

// test conv3d
TEST(SOFIE, Conv3d_B1)
{
   TestConv("3d", 1, false, 3, 2, 3, 1, 0);
}

// Tets recurrent network 
// test with avg pooling
TEST(SOFIE, RNN_B1)
{
   TestRecurrent("RNN", 1, 3, 5, 4, 1);
}

TEST(SOFIE, LSTM_B1)
{
   TestRecurrent("LSTM", 1, 3, 5, 4, 1);
}

TEST(SOFIE, GRU_B1)
{
   TestRecurrent("GRU", 1, 3, 5, 4, 1);
}