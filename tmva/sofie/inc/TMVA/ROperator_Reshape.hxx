#ifndef TMVA_SOFIE_ROPERATOR_RESHAPE
#define TMVA_SOFIE_ROPERATOR_RESHAPE

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <cassert>
#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum ReshapeOpMode { Reshape, Flatten, Squeeze, Unsqueeze };


class ROperator_Reshape final : public ROperator
{

private:

   bool fVerbose = false;
   bool fDimInput = false;
   bool fDynamicShape = false;
   ReshapeOpMode fOpMode = Reshape;   // type of Reshape operator

   int fAllowZero = 0; // (for Reshape) zero in tensor shape makes output shape equal to input tensor shape
   int fAxis = 1;      // (for Flatten)

   std::string fNData;        // input data tensor name
   std::string fNShape;       // reshape tensor name
   std::string fNOutput;               // output tensor name
   std::vector<Dim> fShapeInput;     // input shape data
   std::vector<Dim> fShapeOutput;   // output shape data
   std::vector<int64_t> fAttrAxes;         // axes attributes (provided for all version of Squeeze/Unsqueeze)
   std::vector<int64_t> fShape;     // shape tensor values provided for Reshape

public:

   std::string Name() const {
      if (fOpMode == Reshape) return "Reshape";
      if (fOpMode == Flatten) return "Flatten";
      if (fOpMode == Squeeze) return "Squeeze";
      if (fOpMode == Unsqueeze) return "Unsqueeze";
      return "";
   }

   ROperator_Reshape(){}
   ROperator_Reshape(ReshapeOpMode opMode, int attr_value, std::string nameData, std::string nameShape, std::string nameOutput)
      : fOpMode(opMode), fNData(UTILITY::Clean_name(nameData)), fNShape(UTILITY::Clean_name(nameShape)),
      fNOutput(UTILITY::Clean_name(nameOutput))
   {
      if (opMode == Reshape) fAllowZero = attr_value;
      if (opMode == Flatten) fAxis = attr_value;

      fInputTensorNames = { fNData };
      if(!fNShape.empty()){
         fInputTensorNames.emplace_back(fNShape);
      }
      fOutputTensorNames = { fNOutput };
   }

   // for squeeze/unsqueezed operators following old ONNX version (< 10)
   // In this cases axes are passed as attribute values
   ROperator_Reshape(ReshapeOpMode opMode, std::vector<int64_t> attrAxes, std::string nameData, std::string nameOutput)
      : fOpMode(opMode), fNData(UTILITY::Clean_name(nameData)), fNOutput(UTILITY::Clean_name(nameOutput)),
        fAttrAxes(attrAxes)
   {
      assert(fOpMode == Squeeze || fOpMode == Unsqueeze);
   }

   // output type is same as input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      auto ret = std::vector<ETensorType>(1, input[0]);
      return ret;
   }
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      return input;
   }

   // output shape
   std::vector<std::vector<Dim>> ShapeInference(const std::vector<std::vector<Dim>> & input)  {
      std::vector<std::vector<Dim>> ret;
      auto & input_shape = input[0];

      if (fOpMode == Reshape) {
         // correct the provided shape (here we have the value) for 0 or -1
         std::vector<Dim> output_shape(fShape.size());
         assert(!fShape.empty() && !fDynamicShape);
         for (size_t i = 0; i < output_shape.size(); i++) {
            if (fShape[i] > 0 || (fAllowZero && fShape[i] >= 0))
               output_shape[i] = Dim{ static_cast<size_t>(fShape[i]) };
            else if (!fAllowZero && fShape[i] == 0)
               output_shape[i] = input_shape[i];
         }
         // now case of -1 in shape
         for (size_t i = 0; i < output_shape.size(); i++) {
            if (fShape[i] == -1) {
               auto tmp = output_shape;
               tmp.erase(tmp.begin() + i);
               auto tmp_length = ConvertDimShapeToLength(tmp);
               auto input_length = ConvertDimShapeToLength(input_shape);
               if (IsInteger(tmp_length) && IsInteger(input_length))
                  output_shape[i] = Dim{static_cast<size_t>(std::stoi(input_length) / std::stoi(tmp_length))};
               else
                  output_shape[i] = Dim{std::string("(") + input_length + " / (" + tmp_length + ")"};

               break; // cannot have more than -1
            }
            //  throw std::runtime_error(
            //                   "TMVA Reshape Op : output shape has multiple negative or zero values");
         }

         if (fVerbose)
            std::cout << "Reshape: correct output shape  to " << ConvertShapeToString(output_shape) << std::endl;

         if (!fDimInput && ConvertDimShapeToLength(output_shape) != ConvertDimShapeToLength(input_shape)) {
            throw std::runtime_error("TMVA Reshape Op : Invalid  shapes : " + ConvertShapeToString(input_shape) +
                                     ConvertShapeToString(output_shape));
         }
         ret.push_back(output_shape);

      } else if (fOpMode == Flatten) {
         // flatten case
         if (fAxis < 0)
            fAxis += input_shape.size();
         auto s1 = std::vector<Dim>(input_shape.begin(), input_shape.begin() + fAxis);
         auto s2 = std::vector<Dim>(input_shape.begin() + fAxis, input_shape.end());
         auto l1 = ConvertDimShapeToLength(s1);
         auto l2 = ConvertDimShapeToLength(s2);
         std::vector<Dim> newShape = {Dim{l1}, Dim{l2}};
         ret.push_back(newShape);
      } else if (fOpMode == Squeeze) {
         // squeeze
         // assume no axis is provided - remove all axes with value equal to 1
         auto output_shape = input_shape;
         if (fAttrAxes.empty()) {
            size_t i = 0;
            while (i < output_shape.size()) {
               if (output_shape[i] == Dim{1}) {
                  output_shape.erase(output_shape.begin() + i);
               } else {
                  i++;
               }
            }
         } else {
            auto &axes = fAttrAxes;
            for (size_t i = 0; i < axes.size(); i++) {
               if (axes[i] < 0)
                  axes[i] += input_shape.size();
               if (!(output_shape[axes[i]] == Dim{1}))
                  throw std::runtime_error("TMVA Squeeze Op : Invalid  axis value " + std::to_string(axes[i]) +
                                           " for " + ConvertShapeToString(output_shape));
               output_shape.erase(output_shape.begin() + axes[i]);
            }
         }
         ret.push_back(output_shape);
      }

      else if (fOpMode == Unsqueeze) {
         // unsqueeze
         assert(!fAttrAxes.empty());
         auto output_shape = input_shape;
         auto &axes = fAttrAxes;
         // output rank
         int64_t r = input[0].size() + axes.size();
         for (auto &a : axes) {
            int64_t i = static_cast<int64_t>(a);
            if (i < -r || i > r - 1)
               throw std::runtime_error("TMVA Unsqueeze Op - axes input is not in correct range");
            if (i >= 0)
               output_shape.insert(output_shape.begin() + i, Dim{1});
            else
               // negative axes
               output_shape.insert(output_shape.end() + i + 1, Dim{1});
         }
         ret.push_back(output_shape);
      }
      return ret;
   }

   void Initialize(RModel& model) override {

      fVerbose = model.Verbose();
      if (model.CheckIfTensorAlreadyExist(fNData) == false) {
          // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA Reshape Op Input Tensor " + fNData + "  is not found in model");
      }
      fShapeInput = model.GetDimTensorShape(fNData);
      fDimInput = model.IsDynamicTensor(fNData);
      // check if optional shape tensor exist
      if (!fNShape.empty()) {
         if (model.CheckIfTensorAlreadyExist(fNShape)) {
            if (model.IsConstantTensor(fNShape) || model.IsInitializedTensor(fNShape)) {
               // assume input shape is an initialized tensor
               auto dptr = model.GetInitializedTensorData(fNShape);
               auto input_shape = static_cast<int64_t *>(dptr.get());
               auto vec = model.GetTensorShape(fNShape);
               size_t n = 1;
               if (vec.size() > 0)
                  n = vec[0]; // size of shape input tensor
               // copy values in fShape vector
               std::copy(input_shape, input_shape + n, fShape.begin());
               fShapeOutput = ShapeInference({fShapeInput})[0];
               // set flag to not write tensor in weight file. Its data will be hard-coded in way model is constructed
               model.SetNotWritableInitializedTensor(fNShape);
            } else {
               // we cannot get shape at initialization time but at run-time
               fDynamicShape = true;
               fShapeOutput.resize(fShapeInput.size());
               for (size_t i = 0; i < fShapeInput.size(); i++) {
                  fShapeOutput[i] = Dim{ std::string("s_") + fNOutput + "_" + std::to_string(i)};
               }
            }
         } else {
            throw std::runtime_error("TMVA Reshape Op Shape Tensor " + fNShape + " is not found in model");
         }
      } else if (!fAttrAxes.empty()) {
         // case fNShape is empty and axes are provided as attributes
         fShapeOutput = ShapeInference({fShapeInput})[0];
      } else if (fOpMode == Flatten || fOpMode == Squeeze) {
         fShapeOutput = ShapeInference({fShapeInput})[0];
      } else {
         throw std::runtime_error("TMVA Reshape Op : Invalid Input/Attribute data");
      }
      // check if output is constant or not
      if (model.IsInitializedTensor(fNData) && model.GetTensorType(fNData) == ETensorType::INT64) {
         fIsOutputConstant = true;
         auto inputData = static_cast<int64_t*>(model.GetInitializedTensorData(fNData).get());
         auto o_shape = ConvertShapeToInt(fShapeOutput);
         if (ConvertShapeToLength(ConvertShapeToInt(fShapeInput)) != ConvertShapeToLength(o_shape) )
            throw std::runtime_error("TMVA Reshape Op : Invalid Input/Output lengths");
         model.AddConstantTensor<int64_t>(fNOutput, o_shape, inputData);
         if (model.Verbose()) {
            std::cout << Name() << " : " << fNData << " " << ConvertShapeToString(fShapeInput) << " -->  " << fNOutput << " (constant) " << ConvertShapeToString(fShapeOutput)  << " : " <<
            ConvertValuesToString(ConvertShapeToLength(o_shape), inputData) << std::endl;
         }
      } else {
         // non-constant case
         model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNData), fShapeOutput);
         if (model.Verbose())
            std::cout << Name() << " : " << fNData << " " << ConvertShapeToString(fShapeInput) << " -->  "<< fNOutput << "  " << ConvertShapeToString(fShapeOutput)  << std::endl;
      }
   }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) return "";  //no op for constant tensors

      std::stringstream out;
      std::string opType = "Reshape";
      if (fOpMode == Flatten)
         opType = "Flatten";
      else if (fOpMode == Squeeze)
         opType = "Squeeze";
      else if (fOpMode == Unsqueeze)
         opType = "Unsquueze";

      out << SP << "///--------" << opType << " operator " << opName << "\n" << std::endl;

      // in case of dynamic output shape we need to set the shape value from input shape tensor
      // and take case of the zero values
      if (fDynamicShape) {
         for (size_t i = 0; i < fShapeOutput.size(); i++) {
            out << SP << "int64_t " << fShapeOutput[i].param << " = " << "tensor_" << fNShape << "[" << i << "];\n";
            if (!fAllowZero)
               out << "if (" << fShapeOutput[i].param << " <= 0) " <<  fShapeOutput[i].param << " = " <<  fShapeInput[i] << ";\n";
         }
      }

      // output of reshape is same as input
      auto lengthOut = ConvertDimShapeToLength(fShapeOutput);
      auto lengthIn = ConvertDimShapeToLength(fShapeInput);
      if (lengthOut != lengthIn) {
         // check needs to be done at run-time
         out << SP << "if (" << lengthOut << "!=" << lengthIn << ")\n";
         out << "throw std::runtime_error(\"TMVA SOFIE Reshape Op : wrong output shape - is "
               << ConvertShapeToString(fShapeOutput) << " and input is "
               << ConvertShapeToString(fShapeInput) << ");\n";
      }


      out << SP << "std::copy( tensor_" << fNData << ", tensor_" << fNData << " + " << lengthIn << ", " << "tensor_" << fNOutput
          << ");\n";
      return out.str();
   }
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_RESHAPE
