#include "TMVA/SOFIE/RModelParser_ONNX.hxx"
#include <iostream>

int main() {

   std::cout << "Running GELU parser test..." << std::endl;

   TMVA::SOFIE::RModelParser_ONNX parser;

   try {
      auto model = parser.Parse("gelu.onnx");
      std::cout << "Parsed successfully (unexpected)" << std::endl;
   }
   catch (...) {
      std::cout << "Failed to parse GELU (expected behavior)" << std::endl;
   }

   return 0;
}
