#ifndef TMVA_SOFIE_ROPERATOR_BASICNARY
#define TMVA_SOFIE_ROPERATOR_BASICNARY

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <vector>
#include <sstream>
#include <algorithm>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum class EBasicNaryOperator {Max, Min, Mean, Sum};

template<typename T, EBasicNaryOperator Op>
struct NaryOperatorTraits {};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Max> {
   static const std::string Name() {return "Max";}
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "\t" << "\t" << res << " = " << inputs[0] << ";\n";
      for (size_t i = 1; i < inputs.size(); i++) {
         out << "\t" << "\t" << res << " = std::max(" << res << ", " << inputs[i] << ");\n";
      }
      return out.str();
   }
};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Min> {
   static const std::string Name() {return "Min";}
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "\t" << "\t" << res << " = " << inputs[0] << ";\n";
      for (size_t i = 1; i < inputs.size(); i++) {
         out << "\t" << "\t" << res << " = std::min(" << res << ", " << inputs[i] << ");\n";
      }
      return out.str();
   }
};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Mean> {};

template<>
struct NaryOperatorTraits<float, EBasicNaryOperator::Mean> {
   static const std::string Name() {return "Mean";}
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "\t" << "\t" << res << " = (" << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << " + " << inputs[i];
      }
      out << ") / float(" << inputs.size() << ");\n";
      return out.str();
   }
};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Sum> {
   static const std::string Name() {return "Sum";}
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "\t" << "\t" << res << " = " << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << " + " << inputs[i];
      }
      out << ";\n";
      return out.str();
   }
};

template <typename T, EBasicNaryOperator Op>
class ROperator_BasicNary final : public ROperator
{

private:

   std::vector<std::string> fNInputs;
   std::string fNY;
   std::vector<std::vector<size_t>> fShapeInputs;

   std::vector<std::string> fNBroadcastedInputs;
   std::vector<size_t> fShapeY;

   bool fBroadcast = false;

   std::string fType;

public:
   ROperator_BasicNary(){}

   ROperator_BasicNary( const std::vector<std::string> & inputNames, const std::string& nameY):
   fNY(UTILITY::Clean_name(nameY)){
      fNInputs.reserve(inputNames.size());
      for (auto & name : inputNames)
         fNInputs.push_back(UTILITY::Clean_name(name));

      fInputTensorNames.resize(fNInputs.size());
      std::transform(fNInputs.begin(), fNInputs.end(), fInputTensorNames.begin(),
                  [](const std::string& s) -> std::string { return s; });
      fOutputTensorNames = { fNY };
   }

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = std::vector<std::vector<size_t>>(1, input[0]);
      return ret;
   }

   void Initialize(RModel& model) override {
      for (auto &it : fNInputs) {
         if (!model.CheckIfTensorAlreadyExist(it)) {
            throw std::runtime_error("TMVA SOFIE BasicNary Op Input Tensor " + it + " is not found in model");
         }
         fShapeInputs.push_back(model.GetTensorShape(it));
      }
      // Find the common shape of the input tensors
      fShapeY = UTILITY::MultidirectionalBroadcastShape(fShapeInputs);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInputs[0]), fShapeY);
      // Broadcasting
      size_t N = fNInputs.size();
      fNBroadcastedInputs.reserve(N);
      for (size_t i = 0; i < N; i++) {
         if (!UTILITY::AreSameShape(model.GetTensorShape(fNInputs[i]), fShapeY)) {
            fBroadcast = true;
            std::string name = "Broadcasted"  + fNInputs[i];
            model.AddIntermediateTensor(name, model.GetTensorType(fNInputs[0]), fShapeY);
            fNBroadcastedInputs.emplace_back("tensor_" + name);
            fOutputTensorNames.emplace_back(name);
         } else {
            fNBroadcastedInputs.emplace_back("tensor_" + fNInputs[i]);
         }
      }
      fType = ConvertTypeToString(model.GetTensorType(fNInputs[0]));
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE BasicNary called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShapeY);
      out << SP << "\n//------ BasicNary operator\n";
      if (fBroadcast) {
         for (size_t i = 0; i < fNInputs.size(); i++) {
            if (fNBroadcastedInputs[i] != fNInputs[i]) {
               out << SP << SP << "// Broadcasting " << fNInputs[i] << " to " << ConvertShapeToString(fShapeY) << "\n";
               out << SP << SP << "{\n";
               out << SP << SP << SP << fType << "* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << fType << ">(tensor_" + fNInputs[i] << ", " << ConvertShapeToString(fShapeInputs[i]);
               out << ", " << ConvertShapeToString(fShapeY) << ");\n";
               out << SP << SP << SP << "std::copy(data, data + " << length << ", " << fNBroadcastedInputs[i] << ");\n";
               out << SP << SP << SP << "delete[] data;\n";
               out << SP << SP << "}\n";
            }
         }
      }

      if (fNInputs.size() == 1) {
         out << SP << "std::copy(tensor_" << fNInputs[0] << ", tensor_" << fNInputs[0] << " + ";
         out << length << ", tensor_" << fNY << ");\n";
      } else {
         std::vector<std::string> inputs(fNBroadcastedInputs.size());
         for (size_t i = 0; i < fNBroadcastedInputs.size(); i++) {
            inputs[i] = fNBroadcastedInputs[i] + "[id]";
         }
         out << SP << "for (size_t id = 0; id < " << length << "; id++) {\n";
         out << NaryOperatorTraits<T,Op>::Op("tensor_" + fNY + "[id]", inputs);
         out << SP << "}\n";
      }
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {return { std::string("cmath") }; }
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_BasicNary
