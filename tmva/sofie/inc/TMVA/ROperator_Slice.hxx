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
   std::vector<Dim> fShapeInput;     // input shape data
   std::vector<Dim> fShapeOutput;   // output shape data
   // saved Start/End.Steps are corrected from initial ONNX for negative/default values
   // and are available for each axis
   std::vector<Dim> fStart;         // starting values of slices for all axes
   std::vector<Dim> fEnd;           // End values of slices for all axes
   std::vector<Dim> fSteps;         // step values of slices for all axes
   std::vector<Dim> fStartDims;         // input starting values of slices
   std::vector<Dim> fEndDims;           // input End values of slices
   std::vector<Dim> fStepDims;         // input step values of slices
   std::vector<IType> fAxes;           // axes for input start/emd/step values

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



   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNData) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA Slice Op Input Tensor is not found in model");
      }

      std::vector<std::vector<Dim>> shapes;
      fShapeInput = model.GetDimTensorShape(fNData);
      shapes.push_back(fShapeInput);

      std::vector<std::vector<IType>> itensors(4);

      if (fNames.size() > 0) {  // size has to be equal to 4
         // loop on the extra 2 or 3 or 4 inputs
         for (size_t i = 0; i < 4; ++i) {
            if (!fNames[i].empty()) {
               if (model.IsInitializedTensor(fNames[i])) {
                  // std::cout << " i " << i << " getting data for tensor " << fNames[i] << std::endl;
                  auto dptr = model.GetInitializedTensorData(fNames[i]);
                  auto tensor = static_cast<IType *>(dptr.get());
                  auto vec = model.GetTensorShape(fNames[i]);
                  assert(vec.size() == 1);
                  itensors[i] = std::vector<IType>(tensor, tensor + vec[0]);
               } else {
                  // case is an intermediate tensor
                  auto shape = model.GetTensorShape(fNames[i]);
                  size_t s = shape[0];
                  if (i == 0) {
                     fStartDims = std::vector<Dim>(s);
                  } else if (i == 1) {
                     fEndDims = std::vector<Dim>(s);
                  } else if (i == 3) {
                     fStepDims = std::vector<Dim>(s);
                  }
               }
            }
         }
      } else {
         // old slice versions
         assert(fAttributes.size() > 1);
         for (size_t i = 0; i < fAttributes.size(); i++) {
            itensors[i] = fAttributes[i];
         }
      }
      size_t dim = fShapeInput.size();

      // default values
      fSteps = std::vector<Dim>(dim, Dim{1});
      fStart = std::vector<Dim>(dim, Dim{0});
      fEnd = fShapeInput;

      // default axes
      if (itensors[2].empty()) {
         fAxes.resize(dim);
         std::iota(fAxes.begin(), fAxes.end(), 0);
      } else {
         fAxes = itensors[2];
         for (size_t i = 0; i < fAxes.size(); i++) {
            // negative axes - they count from the back
            if (fAxes[i] < 0) fAxes[i] = dim + fAxes[i];
            if (fAxes[i] < 0 || fAxes[i] >= static_cast<IType>(dim))
               throw std::runtime_error("TMVA Slice Op : invalid axis value " + std::to_string(fAxes[i]) +
                  " for  " + std::to_string(i));
         }
      }
      // if we know the input shape in given axis
      std::cout << "axis loop....\n";
      for (size_t i = 0; i < fAxes.size(); i++) {
         if (!fShapeInput[fAxes[i]].isParam) {
            std::cout << i << " Non param dim for " << fAxes[i] << "  " << fShapeInput[fAxes[i]] << std::endl;
            size_t iAxisDim = fShapeInput[fAxes[i]].dim;
            //correct values if too large or too small
            IType istart = 0;
            if (!itensors[0].empty()) {
               istart = itensors[0][i];
               if (istart < 0) istart = iAxisDim + istart;
               if (istart < 0) istart = 0;
            }
            IType iend = static_cast<IType>(iAxisDim);
            if (!itensors[1].empty()) {
               iend = itensors[1][i];
               if (iend < 0) iend = iAxisDim + iend;
            }
            //steps
            IType istep = 1;
            if (!itensors[3].empty()) {
               istep = itensors[3][i];
            }
            if (istep > 0) {
               if (istart > static_cast<IType>(iAxisDim)) istart = static_cast<IType>(iAxisDim);
               if (iend < 0) iend = 0;
               if (iend > static_cast<IType>(iAxisDim)) iend = static_cast<IType>(iAxisDim);
            } else if (istep < 0) {
               if (istart > static_cast<IType>(iAxisDim)-1) istart = static_cast<IType>(iAxisDim) -1;
               if (iend < -1) iend = -1;
               if (iend > static_cast<IType>(iAxisDim)-1) iend = static_cast<IType>(iAxisDim) -1;
            } else {
               throw std::runtime_error("TMVA Slice Op : invalid step value " + std::to_string(istep) +
                  " for  " + std::to_string(i));
            }
            fStart[fAxes[i]] = Dim{size_t(istart)};
            fEnd[fAxes[i]] = Dim{size_t(iend)};
            fSteps[fAxes[i]] = Dim{size_t(istep)};
         }
         else {
            std::cout << i << " Param dim for " << fAxes[i] << "  " <<  fShapeInput[fAxes[i]] << std::endl;
            // we need to correct at run time
            if (!itensors[0].empty()) {
               IType istart = itensors[0][i];
               if (istart < 0) {
                  std::string sstart = std::string("(") + fShapeInput[fAxes[i]].param + "-" + std::to_string(-istart) +")";
                  fStart[fAxes[i]] = Dim{sstart,size_t(-1)};
               } else {
                 fStart[fAxes[i]] = Dim{size_t(istart)};
               }
            }
            if (!itensors[1].empty()) {
               IType iend = itensors[1][i];
               if (iend < 0) {
                  std::string send = std::string("(") + fShapeInput[fAxes[i]].param + "-" + std::to_string(-iend) +")";
                  fEnd[fAxes[i]] = Dim{send,size_t(-1)};
               } else {
                 fEnd[fAxes[i]] = Dim{size_t(iend)};
               }
            }
            if (!itensors[3].empty()) {
               fSteps[fAxes[i]] = Dim{size_t(itensors[3][i])};
            }
            // case of intermediate tensors for start/end/steps
            if (!fStartDims.empty()) {
               fStartDims[i] = Dim{std::string("start_") + fNOutput + "_" + std::to_string(i)};
               fStart[fAxes[i]] = fStartDims[i];
            }
            if (!fEndDims.empty()) {
               fEndDims[i] = Dim{std::string("end_") + fNOutput + "_" + std::to_string(i)};
               fEnd[fAxes[i]] = fEndDims[i];
            }
            if (!fStepDims.empty()) {
               fStepDims[i] = Dim{std::string("step_") + fNOutput + "_" + std::to_string(i)};
               fSteps[fAxes[i]] = fStepDims[i];
            }
         }



      }
      std::cout << "found output shape " << std::endl;
      //  find output shape
      fShapeOutput.resize(dim);
      for (size_t i = 0; i < dim; i++) {
         if (!fEnd[i].isParam && !fStart[i].isParam && !fSteps[i].isParam) {
            size_t s = (fEnd[i].dim-fStart[i].dim)/ fSteps[i].dim;
            fShapeOutput[i] = Dim{s};
         } else {
            std::string s;
            if (fStart[i].GetVal() != "0")
               s = "(" + fEnd[i].GetVal() + "-" + fStart[i].GetVal() + ")";
            else
               s = fEnd[i].GetVal();
            if (fSteps[i].GetVal() != "1") {
               s.insert(0,"(");
               s += ")/" + fSteps[i].GetVal() + ")";
            }
            fShapeOutput[i] = Dim{s,size_t(-1)};
         }
      }
      // case input is a constant tensor and of int64 type
      if (model.IsInitializedTensor(fNData) && model.GetTensorType(fNData) == ETensorType::INT64) {
         fIsOutputConstant = true;
         auto inputData = static_cast<int64_t*>(model.GetInitializedTensorData(fNData).get());
         size_t outputSize = ConvertShapeToLength(ConvertShapeToInt(fShapeOutput));
         std::vector<int64_t> outputData(outputSize);
         std::vector<size_t> inputStride = UTILITY::ComputeStrideFromShape(ConvertShapeToInt(fShapeInput));
         // perform slice using a recursive function- need to use two lambda functions for this
         auto sliceRecursive = [&](size_t iaxis, size_t & outIdx, size_t & inOffset) {
            auto slice_impl = [&](size_t iax, size_t & outputIdx, size_t & inputOffset, auto & sliceRecImpl) {
               if (fStart[iax].isParam || fEnd[iax].isParam || fSteps[iax].isParam)
                  throw std::runtime_error("TMVA Slice Op : cannot have parametric values hen input is constant");
               // compute indices
               std::vector<IType> indices;
               for (IType i = (IType) fStart[iax].dim; (IType(fSteps[iax].dim) > 0) ? i < IType(fEnd[iax].dim) : i > IType(fEnd[iax].dim); i += IType(fSteps[iax].dim) )
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

         model.AddConstantTensor<int64_t>(fNOutput, ConvertShapeToInt(fShapeOutput), outputData.data());
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
      auto strides = UTILITY::ComputeStrideFromShape(fShapeInput);


      out << SP << "{\n"; // define operator scope
      for (size_t i = 0; i < fStartDims.size(); i++) {
         out << SP << "size_t " << fStartDims[i] << " = tensor_" << fNames[0] << "[" << i << "];\n";
         out << SP << "if (" << fStartDims[i] << " < 0) " << fStartDims[i] << " += " << fShapeInput[fAxes[i]] <<";\n";
      }
      for (size_t i = 0; i < fEndDims.size(); i++) {
         out << SP << "size_t " << fEndDims[i] << " = tensor_" << fNames[1] << "[" << i << "];\n";
         out << SP << "if (" << fEndDims[i] << " < 0) " << fEndDims[i] << " += " << fShapeInput[fAxes[i]] <<";\n";
      }
      for (size_t i = 0; i < fStepDims.size(); i++) {
         // to do : apply correction here for steps too
         out << SP << "size_t " << fStepDims[i] << " = tensor_" << fNames[3] << "[" << i << "];\n";
      }
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
