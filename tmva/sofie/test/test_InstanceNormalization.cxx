// @(#)root/tmva/sofie:$Id$

#include "TMVA/ROperator_InstanceNormalization.hxx"
#include "TMVA/RModel.hxx"
#include <iostream>
#include <vector>
#include <string>
#include <memory>

using namespace TMVA::Experimental::SOFIE;

int main()
{
   std::cout << "Running InstanceNormalization Test..." << std::endl;

   RModel model;
   std::vector<size_t> inputShape = {1, 2, 2, 2};
   std::vector<size_t> paramShape = {2};

   model.AddIntermediateTensor("Input", ETensorType::FLOAT, inputShape);
   model.AddIntermediateTensor("Scale", ETensorType::FLOAT, paramShape);
   model.AddIntermediateTensor("Bias", ETensorType::FLOAT, paramShape);

   auto op = std::make_shared<ROperator_InstanceNormalization<float>>(1e-5, "Input", "Scale", "Bias", "Output");

   op->Initialize(model);
   std::string code = op->Generate("InstanceNormTest");

   bool hasMean = code.find("float mean = sum") != std::string::npos;
   bool hasVar = code.find("float var =") != std::string::npos;

   if (hasMean && hasVar) {
      std::cout << "[SUCCESS] Code generation contains expected logic." << std::endl;
      return 0; // Return 0 tells CMake the test passed
   } else {
      std::cerr << "[FAIL] Generated code is missing logic." << std::endl;
      return 1; // Return 1 tells CMake the test failed
   }
}
