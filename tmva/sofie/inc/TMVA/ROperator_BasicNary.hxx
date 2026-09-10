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
   static std::string Expr(const std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "std::max({ " << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << ", " << inputs[i];
      }
      out << "})";
      return out.str();
   }
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      return res + " = " + Expr(inputs) + ";\n";
   }
   static size_t Func(const std::vector<size_t>& values) {
      return *std::max_element(values.begin(), values.end());
   }
};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Min> {
   static const std::string Name() {return "Min";}
   static std::string Expr(const std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "std::min({ " << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << ", " << inputs[i];
      }
      out << "})";
      return out.str();
   }
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      return res + " = " + Expr(inputs) + ";\n";
   }
   static size_t Func(const std::vector<size_t>& values) {
      return *std::min_element(values.begin(), values.end());
   }
};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Mean> {};

template<>
struct NaryOperatorTraits<float, EBasicNaryOperator::Mean> {
   static const std::string Name() {return "Mean";}
   static std::string Expr(const std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "((" << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << " + " << inputs[i];
      }
      out << ") / float(" << inputs.size() << "))";
      return out.str();
   }
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      return res + " = " + Expr(inputs) + ";\n";
   }
   static size_t Func(const std::vector<size_t>& values) {
      size_t sum = 0;
      for (auto & v : values) sum += v;
      return sum / values.size();
   }
};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Sum> {
   static const std::string Name() {return "Sum";}
   static std::string Expr(const std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "(" << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << " + " << inputs[i];
      }
      out << ")";
      return out.str();
   }
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      return res + " = " + Expr(inputs) + ";\n";
   }
   static size_t Func(const std::vector<size_t>& values) {
      size_t sum = 0;
      for (auto & v : values) sum += v;
      return sum;
   }
};

template <typename T, EBasicNaryOperator Op>
class ROperator_BasicNary final : public ROperator
{

private:

   std::vector<std::string> fNInputs;
   std::string fNY;
   std::vector<std::vector<Dim>> fShapeInputs;

   std::vector<std::string> fNBroadcastedInputs;
   std::vector<size_t> fShapeY;
   std::vector<Dim> fDimShapeY;

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
                  [](const std::string& s) -> std::string_view { return s; });
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

   // Case where all inputs are rank <= 1 INT64 tensors known at initialization time and at least one of
   // them is a shape tensor. The output is then also a shape tensor (its values, possibly symbolic, are
   // computed here) and no code needs to be generated for this operator.
   bool InitializeShapeTensorOutput(RModel &model) {
      bool hasShapeTensor = false;
      bool isScalar = true;
      size_t length = 1;
      for (auto &name : fNInputs) {
         if (model.GetTensorType(name) != ETensorType::INT64)
            return false;
         if (!model.IsShapeTensor(name) && !model.IsInitializedTensor(name))
            return false;
         hasShapeTensor |= model.IsShapeTensor(name);
         auto shape = model.GetTensorShape(name);
         if (shape.size() > 1)
            return false;
         if (!shape.empty()) {
            isScalar = false;
            // only scalars or tensors of the same length can be combined here
            if (shape[0] != 1 && length != 1 && shape[0] != length)
               return false;
            length = std::max(length, shape[0]);
         }
      }
      if (!hasShapeTensor)
         return false;

      // collect the values of every input as Dim's, broadcasting the scalars and the size-1 tensors
      std::vector<std::vector<Dim>> values(fNInputs.size(), std::vector<Dim>(length));
      for (size_t i = 0; i < fNInputs.size(); i++) {
         auto &name = fNInputs[i];
         if (model.IsShapeTensor(name)) {
            auto &dims = model.GetShapeTensorValues(name);
            for (size_t j = 0; j < length; j++)
               values[i][j] = (dims.size() == 1) ? dims[0] : dims[j];
         } else {
            auto data = static_cast<int64_t *>(model.GetInitializedTensorData(name).get());
            size_t n = ConvertShapeToLength(model.GetTensorShape(name));
            for (size_t j = 0; j < length; j++)
               values[i][j] = Dim{static_cast<size_t>(data[(n == 1) ? 0 : j])};
            // deliberately not flagged as non-writable: the same initializer may still be
            // read at run time by another operator, and a non-writable tensor gets no
            // member emitted at all. Leaving it costs a few bytes in the weight file.
         }
      }

      std::vector<Dim> outputValues(length);
      for (size_t j = 0; j < length; j++) {
         bool isConstant = true;
         std::vector<size_t> dims(fNInputs.size());
         std::vector<std::string> exprs(fNInputs.size());
         for (size_t i = 0; i < fNInputs.size(); i++) {
            isConstant &= !values[i][j].isParam;
            dims[i] = values[i][j].dim;
            // cast to size_t so that the parametric dimensions and the literals have a common type
            exprs[i] = "size_t(" + values[i][j].GetVal() + ")";
         }
         if (isConstant)
            outputValues[j] = Dim{NaryOperatorTraits<T, Op>::Func(dims)};
         else
            outputValues[j] = Dim{NaryOperatorTraits<T, Op>::Expr(exprs), static_cast<size_t>(-1)};
      }
      model.AddShapeTensor(fNY, outputValues, isScalar);
      fIsOutputConstant = true;
      if (model.Verbose()) {
         std::cout << NaryOperatorTraits<T, Op>::Name() << " : --> " << fNY << " "
                   << ConvertDimShapeToString(outputValues) << " (shape)" << std::endl;
      }
      return true;
   }

   void Initialize(RModel& model) override {
      std::vector<std::vector<size_t>> inputShapes;
      for (auto &it : fNInputs) {
         if (!model.CheckIfTensorAlreadyExist(it)) {
            throw std::runtime_error("TMVA SOFIE BasicNary Op Input Tensor " + it + " is not found in model");
         }
      }
      if (InitializeShapeTensorOutput(model))
         return;
      for (auto &it : fNInputs) {
         fShapeInputs.push_back(model.GetDimTensorShape(it));
         if (fNInputs.size()> 2) {
            if (model.IsDimInputTensor(it))
               throw std::runtime_error("TMVA SOFIE BasicNary : supports only 2 inputs for dynamic tensors");
            else
               inputShapes.push_back(model.GetTensorShape(it));
         }
      }
      // Find the common shape of the input tensors
      if (fShapeInputs.size() > 2 ) {
         // support dynamic tensors now for input list of size=2
         auto shapeY = UTILITY::MultidirectionalBroadcastShape(inputShapes);
         fDimShapeY = ConvertShapeToDim(shapeY);
      } else if (fShapeInputs.size() == 2 ) {
         auto ret  = UTILITY::MultidirectionalBroadcastShape(fShapeInputs[0], fShapeInputs[1]);
         // use same code as in BinaryOperator (need to extend for input sizes > 2)
         fBroadcast = ret.first;
         fDimShapeY = ret.second;
         // case of all parametric shapes and MultiDirectionalBroadcastShape  return the max of the 2
         // need to do before we declare the output tensor shape and the broadcasted ones
         if (ret.first & 4) {
            // check if one of the parameter is an input dimension
            // define function to find this
            auto IsInputDimParam = [&](const std::string &p) {
               auto inputNames = model.GetInputTensorNames();
               for (auto &input : inputNames) {
                  for (auto &i_s : model.GetDimTensorShape(input)) {
                     if (i_s.isParam && i_s.param == p)
                        return true;
                  }
               }
               return false;
            };
            auto & shapeA = fShapeInputs[0];
            auto & shapeB = fShapeInputs[1];
            for (size_t i = 0; i < fDimShapeY.size(); i++) {
               auto &s = fDimShapeY[i];
               if (s.isParam && s.param.find("std::max") != std::string::npos) {
                  if (IsInputDimParam(shapeA[i].param)) {
                     // case dim is 1 we indicate that the input parameter is equal to 1
                     if (shapeA[i].dim != 1)
                        s = shapeA[i];
                     else
                        s = shapeB[i];
                  } else if (IsInputDimParam(shapeB[i].param)) {
                     if (shapeB[i].dim != 1)
                        s = shapeB[i];
                     else
                        s = shapeA[i];
                  }
               }
            }
         }
      } else if  (fShapeInputs.size() == 1 ) {
         fDimShapeY = fShapeInputs[0];
      }
      if (!fShapeY.empty())
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNInputs[0]), fShapeY);
      else
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNInputs[0]), fDimShapeY);


      fType = ConvertTypeToString(model.GetTensorType(fNInputs[0]));

      if (model.Verbose()) {
         std::cout << NaryOperatorTraits<T, Op>::Name() << " : ";
         if (fNInputs.size() == 2)
            std::cout << ConvertDimShapeToString(fShapeInputs[0]) << " , "
                      << ConvertDimShapeToString(fShapeInputs[1]);
         std::cout << " --> " << ConvertDimShapeToString(fDimShapeY) << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant)
         return "";
      OpName = "op_" + OpName;
      if (fDimShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE BasicNary called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDimShapeToLength(fDimShapeY);
      out << SP << "\n//------ BasicNary operator\n";

      int nInputs = fNInputs.size();

      if (nInputs == 1) {
         out << SP << "std::copy(tensor_" << fNInputs[0] << ", tensor_" << fNInputs[0] << " + ";
         out << length << ", tensor_" << fNY << ");\n";
      } else {

         // implement operator without broadcasting, but using loos on all indices
         std::vector<std::vector<Dim>> inputStrides(nInputs);
         for (int i = 0; i < nInputs; i++)
            inputStrides[i] = UTILITY::ComputeStrideFromShape(fShapeInputs[i]);

         auto stridesY = UTILITY::ComputeStrideFromShape(fDimShapeY);

         // make loop on output indices
         std::string compute_idx_Y;
         int nloop = 0;
         if (fDimShapeY.empty() ||
               std::all_of(fDimShapeY.begin(), fDimShapeY.end(), [](Dim d) { return d.dim == 1 || d.GetVal() == "1"; })) {
            compute_idx_Y = "0";
         } else {
            for (size_t i = 0; i < fDimShapeY.size(); ++i) {
               if (fDimShapeY[i].dim != 1 && fDimShapeY[i].GetVal() != "1") {
                  nloop++;
                  for (int j = 0; j < nloop; j++) out << SP;
                  out << "for (size_t idx_" << i << " = 0; idx_" << i << " < " << fDimShapeY[i]
                      << "; ++idx_" << i << "){\n";
                  compute_idx_Y += "idx_" + std::to_string(i);
                  if (stridesY[i].GetVal() != "1")
                     compute_idx_Y += " * " + stridesY[i].GetVal();
                  compute_idx_Y += " + ";
               }
            }
            // remove last 3 characters " + "
            for (int j = 0; j < 3; j++)
               compute_idx_Y.pop_back();
         }
         // find indices for input tensors
         std::vector<std::string> inputs(nInputs);
         for (int ipt = 0; ipt < nInputs; ipt++ ) {
            std::string compute_idx_X;
            auto & shape = fShapeInputs[ipt];
            auto & stride = inputStrides[ipt];
            if (shape.empty() ||
                std::all_of(shape.begin(), shape.end(), [](Dim d) { return d.dim == 1 || d.GetVal() == "1"; })) {
               compute_idx_X = "0";
            } else {
               for (size_t i = 0; i < shape.size(); ++i) {
                  if (shape[i].dim == 1 || shape[i].GetVal() == "1")
                     continue;
                  compute_idx_X += "idx_" + std::to_string(i + (fDimShapeY.size() - shape.size()));
                  if (stride[i].GetVal() != "1")
                     compute_idx_X += " * " + stride[i].GetVal();
                  compute_idx_X += " + ";
               }
               // remove last 3 character " + "
               for (int j = 0; j < 3; j++)
                  compute_idx_X.pop_back();
            }
            inputs[ipt] = "tensor_" + fNInputs[ipt] + "[" + compute_idx_X + "]";
         }

         // perform the operation
         for (int j = 0; j < nloop + 1; j++) out << SP;
         std::string output = "tensor_" + fNY + "[" + compute_idx_Y + "]";
         out << NaryOperatorTraits<T,Op>::Op(output, inputs);

         for (int i = nloop; i > 0; i--) {
            for (int j = 0; j < i; j++) out << SP;
            out << "}\n";
         }
      }
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {return { std::string("cmath") }; }
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_BasicNary
