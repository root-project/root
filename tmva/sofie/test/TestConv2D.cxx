//#include "TPython.h"
#include "TSystem.h"
#include "TMacro.h"
#include <vector>
#include <fstream>
#include <limits>

#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

#include  "gtest/gtest.h"


TEST(SOPHIE,Conv2d)
//void TestConv2D()
{
   int nbatches = 3;
   int ngroups = 2;
   int nlayers = 4;

   // input size is fixed to (nb, 3, 3, 3)
   const int nchannels = 2;
   const int nd = 4;

   const int inputSize = nchannels  * nd * nd;

   //const char *argv[5] = {}
   std::string argv[5];

   argv[0] = std::to_string(nbatches);
   //.c_str();
   argv[1] = std::to_string(nchannels);
   argv[2] = std::to_string(nd);
   argv[3] = std::to_string(ngroups);
   argv[4] = std::to_string(nlayers);
   std::string command = "python Conv2dModelGenerator.py ";
   for (int i = 0; i < 5; i++) {
      command += " ";
      command += argv[i];
   }
   printf("executing %s\n", command.c_str());
   gSystem->Exec(command.c_str());
   // TPython::ExecScript("Conv2dModelGenerator.py",5,argv);

   std::string modelName = "Conv2dModel";
   using namespace TMVA::Experimental;
   SOFIE::RModelParser_ONNX parser;
   std::string inputName = modelName + ".onnx";
   std::cout << "parsing file " << inputName << std::endl;
   SOFIE::RModel model = parser.Parse(inputName);
   std::cout << "generating model.....\n";
   model.Generate(1,1);
   std::string  outputName = modelName + ".hxx";
   std::cout << "writing model as header .....\n";
   model.OutputGenerated();//outputName); 
   std::cout << "output written in  " << outputName << std::endl;

   TMacro m("testSofie"); // inference code;
   m.AddLine(std::string("#include \"" + modelName + ".hxx\"").c_str());
   m.AddLine("std::vector<float> testSofie() {");
   m.AddLine(std::string("   TMVA_SOFIE_" + modelName + "::Session s;").c_str());
   // input must be tensor (1,2,d,d)
   // this is hardcoded for 2 channels
   m.AddLine(std::string("   std::vector<float> input(" + std::to_string(nbatches * inputSize) + ");").c_str());
   m.AddLine(std::string("   for (int ib = 0; ib < " + std::to_string(nbatches) + "; ib++){").c_str());
   m.AddLine(std::string("      std::vector<float> x1(" + std::to_string(nd*nd) + ", float(ib+1));").c_str());
   m.AddLine(std::string("      std::vector<float> x2(" + std::to_string(nd*nd) + ", -float(ib+1));").c_str());
   m.AddLine(std::string("      std::copy(x1.begin(),x1.end(),input.begin() + ib * " + std::to_string(inputSize) + ");").c_str());
   m.AddLine(std::string("      std::copy(x2.begin(),x2.end(),input.begin() + ib * " + std::to_string(inputSize) + " + x1.size());\n   }\n").c_str());

   m.AddLine("   printf(\"doing inference.....\");");
   m.AddLine("   std::vector<float> y = s.infer(input.data());");
   m.AddLine("   return y;}");

   m.Print();

   std::cout << "execute the macro \n";
   
   std::vector<float> * result = (std::vector<float> *)m.Exec();

   // read reference value from test file
   std::vector<float> refValue(result->size());

   std::ifstream f(std::string(modelName + ".out").c_str());
   for (size_t i = 0; i < refValue.size(); ++i) {
      f >> refValue[i];
      std::cout << " result " << result->at(i) << " reference " << refValue[i] << std::endl;
      if (std::abs(refValue[i]) > 0.5)
         EXPECT_FLOAT_EQ(result->at(i), refValue[i]);
      else 
         // expect float fails for small values
         EXPECT_NEAR(result->at(i), refValue[i], 10 * std::numeric_limits<float>::epsilon());
   }
}
