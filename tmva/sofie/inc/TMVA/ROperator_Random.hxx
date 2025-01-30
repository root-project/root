#ifndef TMVA_SOFIE_ROPERATOR_Random
#define TMVA_SOFIE_ROPERATOR_Random

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

   // Random operator for RandomUniform, RandomUniformLike,
   //   RandomNormal, RandomNormalLike
enum RandomOpMode { kUniform, kNormal};

class ROperator_Random final : public ROperator
{
public:

   bool fUseROOT = true; // use ROOT or std for random number generation
private:

   RandomOpMode fMode;
   ETensorType fType;
   std::string fNX;
   std::string fNY;
   int fSeed;
   std::vector<size_t> fShapeY;
   std::map<std::string,float> fParams; // parameter for random generator (e.g. low,high or mean and scale)



public:

   ROperator_Random(){}
   ROperator_Random(RandomOpMode mode, ETensorType type, const std::string & nameX, const std::string & nameY, const std::vector<size_t> & shape, const std::map<std::string, float> & params, float seed) :
      fMode(mode),
      fType(type),
      fNX(UTILITY::Clean_name(nameX)),
      fNY(UTILITY::Clean_name(nameY)),
      fSeed(seed),
      fShapeY(shape),
      fParams(params)
      {
         fInputTensorNames = {  };
         fOutputTensorNames = { fNY };
      }


   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override  {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {

      model.AddIntermediateTensor(fNY, fType, fShapeY);

      if (fUseROOT) {
         model.AddNeededCustomHeader("TRandom3.h");
      }

      // use default values
      if (fMode == kNormal) {
         if (fParams.count("mean") == 0 )
            fParams["mean"] = 0;
         if (fParams.count("scale") == 0)
            fParams["scale"] = 1;
      }
      if (fMode == kUniform) {
         if (fParams.count("low") == 0)
            fParams["low"] = 0;
         if (fParams.count("high") == 0)
            fParams["high"] = 1;
      }

      if (model.Verbose()) {
         std::cout << "Random";
         if (fMode == kNormal) std::cout << "Normal";
         else if (fMode == kUniform) std::cout << "Uniform";
         std::cout << " op  -> " << fNY << " : " << ConvertShapeToString(fShapeY) << std::endl;
         for (auto & p : fParams)
            std::cout << p.first << " : " << p.second << std::endl;
      }
   }
   // generate declaration code for random number generators
   std::string GenerateDeclCode() override {
      std::stringstream out;
      out << "std::unique_ptr<TRandom> fRndmEngine;   // random number engine\n";
      return out.str();
   }
   // generate initialization code for random number generators
   std::string GenerateInitCode() override {
      std::stringstream out;
      out << "//--- creating random number generator ----\n";
      if (fUseROOT) {
         // generate initialization code for creating random number generator
         out << SP << "fRndmEngine.reset(new TRandom3(" << fSeed << "));\n";
      }
      else {
         // not supported
      }
      return out.str();
   }
   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;

      std::stringstream out;
      out << "\n//------ Random";
      if (fMode == kNormal) out << "Normal\n";
      else if (fMode == kUniform) out << "Uniform\n";

      // generate the random array
      int length = ConvertShapeToLength(fShapeY);
      out << SP << "for (int i = 0; i < " << length << "; i++) {\n";
      if (fUseROOT) {
         if (fMode == kNormal) {
            if (fParams.count("mean") == 0 || fParams.count("scale") == 0)
               throw std::runtime_error("TMVA SOFIE RandomNormal op : no mean or scale are defined");
            float mean = fParams["mean"];
            float scale = fParams["scale"];
            out << SP << SP << "tensor_" << fNY << "[i] = fRndmEngine->Gaus(" << mean << "," << scale << ");\n";
         } else if (fMode == kUniform) {
            if (fParams.count("high") == 0 || fParams.count("low") == 0)
              throw std::runtime_error("TMVA SOFIE RandomUniform op : no low or high are defined");
            float high = fParams["high"];
            float low = fParams["low"];
            out << SP << SP << "tensor_" << fNY << "[i] = fRndmEngine->Uniform(" << low << "," << high << ");\n";
         }
      }
      out << SP << "}\n";

      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {
      std::vector<std::string> ret = {"memory"};   // for unique ptr
      return ret;
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Swish
