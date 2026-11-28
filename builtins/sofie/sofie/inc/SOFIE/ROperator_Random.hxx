#ifndef SOFIE_ROPERATOR_Random
#define SOFIE_ROPERATOR_Random

#include "SOFIE/common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

namespace SOFIE{

   // Random operator for RandomUniform, RandomUniformLike,
   //   RandomNormal, RandomNormalLike
enum RandomOpMode { kUniform, kNormal};

class ROperator_Random final : public ROperator
{
public:

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


   void Initialize(RModel& model) override {

      model.AddIntermediateTensor(fNY, fType, fShapeY);

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
      out << "std::unique_ptr<std::mt19937> fRndmEngine;   // random number engine\n";
      return out.str();
   }
   // generate initialization code for random number generators
   std::string GenerateInitCode() override {
      std::stringstream out;
      out << "//--- creating random number generator ----\n";
      // generate initialization code for creating the random number generator
      out << SP << "fRndmEngine.reset(new std::mt19937(" << static_cast<unsigned int>(fSeed) << "u));\n";
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
      if (fMode == kNormal) {
         if (fParams.count("mean") == 0 || fParams.count("scale") == 0)
            throw std::runtime_error("SOFIE RandomNormal op : no mean or scale are defined");
         float mean = fParams["mean"];
         float scale = fParams["scale"];
         out << SP << "std::normal_distribution<float> distribution_" << OpName << "(static_cast<float>(" << mean
             << "), static_cast<float>(" << scale << "));\n";
      } else if (fMode == kUniform) {
         if (fParams.count("high") == 0 || fParams.count("low") == 0)
            throw std::runtime_error("SOFIE RandomUniform op : no low or high are defined");
         float high = fParams["high"];
         float low = fParams["low"];
         out << SP << "std::uniform_real_distribution<float> distribution_" << OpName << "(static_cast<float>(" << low
             << "), static_cast<float>(" << high << "));\n";
      }
      out << SP << "for (int i = 0; i < " << length << "; i++) {\n";
      out << SP << SP << "tensor_" << fNY << "[i] = distribution_" << OpName << "(*this->fRndmEngine);\n";
      out << SP << "}\n";

      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {
      std::vector<std::string> ret = {"memory", "random"};   // for unique ptr, the random number engine and distributions
      return ret;
   }

};

}//SOFIE


#endif //SOFIE_ROPERATOR_Swish
