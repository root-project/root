#ifndef TMVA_SOFIE_ROPERATOR_SLICE
#define TMVA_SOFIE_ROPERATOR_SLICE

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <cassert>
#include <sstream>
#include <numeric>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

// slice operator

template <typename IType>
class ROperator_Slice final : public ROperator
{

private:

   std::string fNData;        // input data tensor name
   std::string fNOutput;      // output data name
   std::vector<std::string> fNames;       // tensor names for meta(axis) information
   std::vector<size_t> fShapeInput;     // input shape data
   std::vector<size_t> fShapeOutput;   // output shape data
   // saved Start/End.Steps are corrected from initial ONNX for negative/default values
   // and are available for each axis
   std::vector<IType> fStart;         // starting values of slices
   std::vector<IType> fEnd;           // End values of slices
   std::vector<IType> fSteps;         // step values of slices

   std::vector<std::vector<IType>> fAttributes; // attributes for the version <=10 case


public:

   ROperator_Slice(){}

   // ctor for versions >= 10
   ROperator_Slice(std::string nameData, std::vector<std::string> names, std::string nameOutput)
      : fNData(UTILITY::Clean_name(nameData)),
      fNOutput(UTILITY::Clean_name(nameOutput))
   {
    fNames.resize(4);
    // axes and steps can be optional
    for (size_t i = 0; i < names.size(); ++i) {
        fNames[i] = UTILITY::Clean_name(names[i]);
    }

    fInputTensorNames = { fNData };
    fOutputTensorNames = { fNOutput };
   }
   // ctor for versions < 10
   ROperator_Slice(std::string nameData, std::vector<IType> starts, std::vector<IType> ends, std::vector<IType> axes, std::string nameOutput)
      : fNData(UTILITY::Clean_name(nameData)),
      fNOutput(UTILITY::Clean_name(nameOutput))
   {
     fAttributes.push_back(starts);
     fAttributes.push_back(ends);
     fAttributes.push_back(axes);
    }

   // output type is same as input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      auto ret = std::vector<ETensorType>(1, input[0]);
      return ret;
   }

   // output shape
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto & input_shape = input[0];
       // assume dimension of output shape is SAME AS INPUT !
      std::vector<std::vector<size_t>> ret(1, input_shape);
      auto & output_shape = ret[0];
      for (size_t i = 0; i < input_shape.size(); i++) {
          output_shape[i] = (fEnd[i]-fStart[i])/ fSteps[i];
      }
      return ret;
   }


   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNData) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA Slice Op Input Tensor is not found in model");
      }

      std::vector<std::vector<size_t>> shapes;
      fShapeInput = model.GetTensorShape(fNData);
      shapes.push_back(fShapeInput);

      std::vector<std::vector<IType>> itensors(4);
      if (fNames.size() > 0) {
         // loop on the extra 2 or 3 or 4 inputs
         for (size_t i = 0; i < fNames.size(); ++i) {
            if (!fNames[i].empty()) {
               // std::cout << " i " << i << " getting data for tensor " << fNames[i] << std::endl;
               auto dptr = model.GetInitializedTensorData(fNames[i]);
               auto tensor = static_cast<IType *>(dptr.get());
               auto vec = model.GetTensorShape(fNames[i]);
               assert(vec.size() == 1);
               itensors[i] = std::vector<IType>(tensor, tensor + vec[0]);
            } else {
               switch (i) {
               case 2: // missing axes
                  itensors[2] = std::vector<IType>(fShapeInput.size());
                  std::iota(itensors[2].begin(), itensors[2].end(), 0);
                  break;
               case 3: // missing steps
                  itensors[3] = std::vector<IType>(itensors[0].size(), 1);
               default: break;
               }
            }
         }
      } else {
         assert(fAttributes.size() > 1);
         for (size_t i = 0; i < fAttributes.size(); i++) {
            itensors[i] = fAttributes[i];
         }
      }
      size_t dim = fShapeInput.size();

      fSteps = std::vector<IType>(dim, 1);
      fStart = std::vector<IType>(dim, 0);
      fEnd = std::vector<IType>(dim, 0);
      std::copy(fShapeInput.begin(), fShapeInput.end(), fEnd.begin());

      auto istart = itensors[0];
      auto iend = itensors[1];
      auto iaxes = itensors[2];
      auto isteps  = itensors[3];

      // make tensor axis
      // if iaxes.size is =0 tensor axis is missing and use defaults
      if (iaxes.size() > 0) {
         for (size_t i = 0; i < iaxes.size(); i++) {
            // negative axes - they count from the back
            if (iaxes[i] < 0) iaxes[i] = dim + iaxes[i];
            if (iaxes[i] < 0 || iaxes[i] >= static_cast<IType>(dim))
               throw std::runtime_error("TMVA Slice Op : invalid axis value " + std::to_string(iaxes[i]) +
                  " for  " + std::to_string(i));

            size_t iAxisDim = fShapeInput[iaxes[i]];
            // find start/end/step for given axis
            // check step size for clamping starting/end value
            if (istart[i] < 0) istart[i] = iAxisDim + istart[i];
            if (iend[i] < 0) iend[i] = iAxisDim + iend[i];
            if (istart[i] < 0) istart[i] = 0;
            if (isteps[i] > 0) {
               if (istart[i] > static_cast<IType>(iAxisDim)) istart[i] = static_cast<IType>(iAxisDim);
               if (iend[i] < 0) iend[i] = 0;
               if (iend[i] > static_cast<IType>(iAxisDim)) iend[i] = static_cast<IType>(iAxisDim);
            } else if (isteps[i] < 0) {
               if (istart[i] > static_cast<IType>(iAxisDim)-1) istart[i] = static_cast<IType>(iAxisDim) -1;
               if (iend[i] < -1) iend[i] = -1;
               if (iend[i] > static_cast<IType>(iAxisDim)-1) iend[i] = static_cast<IType>(iAxisDim) -1;
            } else {
               throw std::runtime_error("TMVA Slice Op : invalid step value " + std::to_string(isteps[i]) +
                  " for  " + std::to_string(i));
            }
            fStart[iaxes[i]] = istart[i];
            fEnd[iaxes[i]] = iend[i];
            fSteps[iaxes[i]] = isteps[i];
         }
      }

      fShapeOutput = ShapeInference({fShapeInput})[0];
      // case input is a constant tensor and of int64 type
      if (model.IsInitializedTensor(fNData) && model.GetTensorType(fNData) == ETensorType::INT64) {
         fIsOutputConstant = true;
         auto inputData = static_cast<int64_t*>(model.GetInitializedTensorData(fNData).get());
         size_t outputSize = ConvertShapeToLength(fShapeOutput);
         std::vector<int64_t> outputData(outputSize);
         std::vector<size_t> inputStride = UTILITY::ComputeStrideFromShape(fShapeInput);
         // perform slice using a recursive function- need to use two lambda functions for this
         auto sliceRecursive = [&](size_t iaxis, size_t & outIdx, size_t & inOffset) {
            auto slice_impl = [&](size_t iax, size_t & outputIdx, size_t & inputOffset, auto & sliceRecImpl) {
               // compute indices
               std::vector<IType> indices;
               for (IType i = fStart[iax]; (fSteps[iax] > 0) ? i < fEnd[iax] : i > fEnd[iax]; i += fSteps[iax] )
                  indices.push_back(i);
               if (iax == dim-1) { // last axis
                  for (size_t i = 0; i < indices.size(); i++) {
                     outputData[outputIdx] = inputData[inputOffset + indices[i]];
                     outputIdx++;
                  }
                  return;
               } else {
                  for (size_t i = 0; i < indices.size(); i++) {
                     size_t offset = inputOffset + inputStride[iax]*indices[i];
                     sliceRecImpl(iax+1, outputIdx, offset,sliceRecImpl);
                  }
               }
            };
            slice_impl(iaxis, outIdx, inOffset,slice_impl);
         };
         size_t idx = 0;
         size_t offset = 0;
         sliceRecursive(0, idx, offset);

         model.AddConstantTensor<int64_t>(fNOutput, fShapeOutput, outputData.data());
         if (model.Verbose()) {
            std::cout << "Slice: output is a constant tensor " << ConvertShapeToString(fShapeOutput) << " : "
                     << ConvertValuesToString(outputData) << std::endl;
         }
      }
      else {
         model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNData), fShapeOutput);
         if (model.Verbose()) {
            std::cout << "Slice ---> " << fNOutput << " " <<  ConvertShapeToString(fShapeOutput) << std::endl;
         }
      }
   }

   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant) return "";  //no op for constant tensors

      OpName = "op_" + OpName;
      if (fShapeInput.empty() || fShapeOutput.empty()){
         throw std::runtime_error("TMVA SOFIE Slice Op called to Generate without being initialized first");
      }

      std::stringstream out;
      //std::string opName = "Slice";

      out << SP << "///------- Slice operator\n" << std::endl;
      // loop on the dimensions depending no the orders
      size_t ndim = fShapeInput.size();
      std::vector<size_t> strides(ndim,1);
      for (int i = int(ndim-2); i >=0 ; i--) {
          strides[i] = strides[i+1]*fShapeInput[i+1];
      }

      out << SP << "{\n"; // define operator scope
      out << SP << "size_t iOut = 0;\n";
      std::string MSP = SP;
      for (size_t idim = 0; idim < ndim; idim++) {
        out << MSP << "for (size_t i" << idim << " = " << fStart[idim] <<  "; i" << idim << " < " << fEnd[idim]
            << "; i" << idim << "+= " << fSteps[idim] << ") {\n";
        MSP += SP;
        if (idim < ndim-1) out << MSP << "size_t stride" << idim << " = " << strides[idim] << "*i" << idim << ";\n";
      }
      out << MSP << "size_t iInput = ";
      for (size_t idim = 0; idim < ndim-1; idim++) out << " stride" << idim << " + ";
      // here should be step size ?
      out << "i" << ndim-1 << ";\n";
      out << MSP << "tensor_" << fNOutput << "[iOut++] = tensor_" <<fNData << "[iInput];\n";
      for (size_t idim = 0; idim < ndim; idim++) {
          MSP = MSP.replace(0,SP.length(),"");
          out << MSP << "}\n";
      }
      out << SP << "}\n"; // end operator scope

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_SLICE
