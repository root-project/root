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

   // flags to indicate if start/end and steps are not defined at compiled time
   bool fIsStartUndef = false;
   bool fIsEndUndef = false;
   bool fIsStepUndef = false;
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
                  auto dptr = model.GetInitializedTensorData(fNames[i]);
                  auto tensor = static_cast<IType *>(dptr.get());
                  auto vec = model.GetTensorShape(fNames[i]);
                  assert(vec.size() == 1);
                  itensors[i] = std::vector<IType>(tensor, tensor + vec[0]);

               } else if (model.IsShapeTensor(fNames[i])) {
                  // case is a shape tensor
                  if (i == 0) {
                     fStartDims = model.GetShapeTensorValues(fNames[i]);
                  } else if (i == 1) {
                     fEndDims = model.GetShapeTensorValues(fNames[i]);
                  } else if (i == 3) {
                     fStepDims = model.GetShapeTensorValues(fNames[i]);
                  }
               } else {
                  // case is an intermediate tensor
                  auto shape = model.GetTensorShape(fNames[i]);
                  size_t s = shape[0];
                  for (size_t k = 0; k < s; k++) {
                     if (i == 0) {
                        fStartDims.push_back( Dim{std::string("start_") + fNOutput + "_" + std::to_string(k)});
                        fIsStartUndef = true;
                     } else if (i == 1) {
                        fEndDims.push_back(Dim{std::string("end_") + fNOutput + "_" + std::to_string(k)});
                        fIsEndUndef = true;
                     } else if (i == 3) {
                        fStepDims.push_back(Dim{std::string("step_") + fNOutput + "_" + std::to_string(k)});
                        fIsStepUndef = true;
                     }
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
      // Loop on axis to get start/end/step values
      for (size_t i = 0; i < fAxes.size(); i++) {
         if (!itensors[0].empty() )
            fStartDims.push_back(Dim{ static_cast<size_t>(itensors[0][i])});
         if (fStartDims.empty())
            throw std::runtime_error("TMVA Slice Op : Missing start input tensor");

         if (!itensors[1].empty())
            fEndDims.push_back(Dim{ static_cast<size_t>(itensors[1][i])});
         else if (fEndDims.empty())
            throw std::runtime_error("TMVA Slice Op : Missing end input tensor");

         if (!itensors[3].empty()) {
            fStepDims.push_back(Dim{ static_cast<size_t>(itensors[3][i])});
         }
         else if (fStepDims.size() < fAxes.size())  // this can happen since it is optional
            fStepDims.push_back(Dim{size_t(1)});

         if (!fShapeInput[fAxes[i]].isParam) {
            size_t iAxisDim = fShapeInput[fAxes[i]].dim;
            //correct values if too large or too small
            IType istart = 0;
            if (!fStartDims[i].isParam) {
               istart = static_cast<IType>(fStartDims[i].dim);
               if (istart < 0) istart = iAxisDim + istart;
            }
            IType iend = static_cast<IType>(iAxisDim);
            if (!fEndDims[i].isParam) {
               iend = static_cast<IType>(fEndDims[i].dim);
               if (iend < 0) iend = iAxisDim + iend;
            }
            //steps
            IType istep = 1;
            if (!fStepDims[i].isParam) {
               istep = static_cast<IType>(fStepDims[i].dim);
            } else {
               throw std::runtime_error("TMVA Slice Op : parametric step inputs are not supported");
            }
            // clamp start end values depending on steps
            // start must be [0,N] for positive steps or [0,N-1] for negative
            // end   must be [0,N] for positive steps or [-1, N-1] for negative
            if (istart < 0) istart = 0;
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
            // for parametric values clamping we will done at run time
            if (fStartDims[i].isParam)
               fStart[fAxes[i]] = fStartDims[i];
            else
               fStart[fAxes[i]] = Dim{size_t(istart)};
            if (fStartDims[i].isParam)
               fEnd[fAxes[i]] = fEndDims[i];
            else
               fEnd[fAxes[i]] = Dim{size_t(iend)};

            fSteps[fAxes[i]] = Dim{size_t(istep)};
         } else {
            //std::cout << i << " Param dim for " << fAxes[i] << "  " <<  fShapeInput[fAxes[i]] << std::endl;
            // correct only negative values
            if (!fStartDims[i].isParam) {
               IType istart = static_cast<IType>(fStartDims[i].dim);
               if (istart < 0) {
                  std::string sstart = std::string("(") + fShapeInput[fAxes[i]].param + "-" + std::to_string(-istart) +")";
                  fStart[fAxes[i]] = Dim{sstart,size_t(-1)};
               } else {
                 fStart[fAxes[i]] = Dim{size_t(istart)};
               }
            } else {
               fStart[fAxes[i]] = fStartDims[i];
            }
            if (!fEndDims[i].isParam) {
               IType iend = static_cast<IType>(fEndDims[i].dim);
               if (iend < 0) {
                  std::string send = std::string("(") + fShapeInput[fAxes[i]].param + "-" + std::to_string(-iend) +")";
                  fEnd[fAxes[i]] = Dim{send,size_t(-1)};
               } else {
                 fEnd[fAxes[i]] = Dim{size_t(iend)};
               }
            } else {
               fEnd[fAxes[i]] = fEndDims[i];
            }

            fSteps[fAxes[i]] = fStepDims[i];
         }

      }
      //  find output shape
      fShapeOutput.resize(dim);
      for (size_t i = 0; i < dim; i++) {
         if (!fEnd[i].isParam && !fStart[i].isParam && !fSteps[i].isParam) {
            int64_t istart = static_cast<int64_t>(fStart[i].dim);
            int64_t iend = static_cast<int64_t>(fEnd[i].dim);
            int64_t istep= static_cast<int64_t>(fSteps[i].dim);
            int64_t s = (iend-istart)/istep;
            fShapeOutput[i] = Dim{static_cast<size_t>(s)};
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
            // add also the shape parameters to RModel to declare them when
            // allocating output tensor
            if (fEnd[i].isParam && fEnd[i].dim != size_t(-1))
               model.AddShapeParam(fEnd[i].param,fEnd[i].dim );
            if (fStart[i].isParam && fStart[i].dim != size_t(-1))
               model.AddShapeParam(fStart[i].param,fStart[i].dim );
            if (fSteps[i].isParam && fSteps[i].dim != size_t(-1))
               model.AddShapeParam(fSteps[i].param,fSteps[i].dim );

         }
      }
      // case input is a constant tensor and of int64 type
      if (model.IsInitializedTensor(fNData) && model.GetTensorType(fNData) == ETensorType::INT64) {
         fIsOutputConstant = true;
         auto inputData = static_cast<int64_t*>(model.GetInitializedTensorData(fNData).get());
         size_t outputSize = ConvertShapeToLength(ConvertShapeToInt(fShapeOutput));
         std::vector<int64_t> outputData(outputSize);
         std::vector<size_t> inputStride = UTILITY::ComputeStrideFromShape(ConvertShapeToInt(fShapeInput));
         std::cout << "slice " << ConvertDimShapeToString(fShapeInput) << " output size " << outputSize << "  " << ConvertDimShapeToString(fShapeOutput) << std::endl;
         std::cout << " start - end -steps \n";
         for (size_t ii = 0; ii< fStart.size(); ii++)
            std::cout << fStart[ii] << "  " << fEnd[ii] << "  " << fSteps[ii] << std::endl;
          // perform slice using a recursive function- need to use two lambda functions for this
         auto sliceRecursive = [&](size_t iaxis, size_t & outIdx, size_t & inOffset) {
            auto slice_impl = [&](size_t iax, size_t & outputIdx, size_t & inputOffset, auto & sliceRecImpl) {
               std::cout << "SLice_impl " << fStart.size() << "  " << fEnd.size() << " " << fSteps.size() << "  " << iax << std::endl;
               if (fStart[iax].isParam || fEnd[iax].isParam || fSteps[iax].isParam)
                  throw std::runtime_error("TMVA Slice Op : cannot have parametric values when input is constant");
               // compute indices
               std::vector<IType> indices;
               for (IType i = (IType) fStart[iax].dim; (IType(fSteps[iax].dim) > 0) ? i < IType(fEnd[iax].dim) : i > IType(fEnd[iax].dim); i += IType(fSteps[iax].dim) )
                  indices.push_back(i);
               if (iax == dim-1) { // last axis
                  std::cout << "SLice_impl last axis: " << indices.size() << " : ";
                  for (size_t i = 0; i < indices.size(); i++) {
                     std::cout << outputIdx << " , " << indices[i] << " " << inputOffset << " ; ";
                     outputData[outputIdx] = inputData[inputOffset + indices[i]];
                     outputIdx++;
                  }
                  std::cout << std::endl;
                  return;
               } else {
                  std::cout << "SLice_impl else : " << indices.size() << " : ";
                  for (size_t i = 0; i < indices.size(); i++) {
                     std::cout << inputStride[iax] << " , " << indices[i] << " " << inputOffset << "  ";
                     size_t offset = inputOffset + inputStride[iax]*indices[i];
                     sliceRecImpl(iax+1, outputIdx, offset,sliceRecImpl);
                  }
                  std::cout << std::endl;
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
      for (size_t i = 0; i < fStepDims.size(); i++) {
         if (fStepDims[i].isParam) {
            if (fIsStepUndef)
               out << SP << "size_t " << fStepDims[i] << " = tensor_" << fNames[3] << "[" << i << "];\n";
         }
      }
      // special case for parametric  values for start/end. Need to do clipping
      for (size_t i = 0; i < fStartDims.size(); i++) {
         if (fStartDims[i].isParam && fStartDims[i].param != fShapeInput[fAxes[i]].param) {
            std::string s_start = "start_" + std::to_string(i);
            if (fIsStartUndef) {
               s_start = fStartDims[i].param;
               out << SP << "size_t " << s_start << " = tensor_" << fNames[0] << "[" << i << "];\n";
            } else {
               out << SP << "size_t " << s_start << " = " <<  fStartDims[i] << ";\n";
               fStart[fAxes[i]] = s_start; // need to use this value later when slicing
            }
            out << SP << "if (" << s_start << " < 0) " << s_start << " += " << fShapeInput[fAxes[i]] <<";\n";
            out << SP << "if (" << s_start << " < 0) " << s_start << " = 0;\n";
            if (!fStepDims[i].isParam) {
               if (static_cast<IType>(fStepDims[i].dim) > 0 )
                  out << SP << "if (" << s_start << " > " << fShapeInput[fAxes[i]] << " ) " << s_start << " = " << fShapeInput[fAxes[i]] <<";\n";
               else
                  out << SP << "if (" << s_start << " > " << fShapeInput[fAxes[i]] << " - 1" << " ) " << s_start << " = " << fShapeInput[fAxes[i]] << " - 1;\n";
            }
         }
         // special case if step is negative and shape are equal and step is negative
         else if (fStartDims[i].isParam && fStartDims[i].param == fShapeInput[fAxes[i]].param && !fStepDims[i].isParam && static_cast<IType>(fStepDims[i].dim) < 0 ) {
            fStart[fAxes[i]] = Dim{ fStartDims[i].param + "-1" };
         }
      }
      // now to for end
      for (size_t i = 0; i < fEndDims.size(); i++) {
         if (fEndDims[i].isParam && fEndDims[i].param != fShapeInput[fAxes[i]].param) {
            std::string s_end = "end_" + std::to_string(i);
            if (fIsEndUndef) {
               s_end = fEndDims[i].param;
               out << SP << "size_t " << s_end << " = tensor_" << fNames[1] << "[" << i << "];\n";
            } else {
               out << SP << "size_t " << s_end << " = " <<  fEndDims[i] << ";\n";
               fEnd[fAxes[i]] = s_end; // need to use this value later when slicing
            }
            out << SP << "if (" << s_end << " < 0) " << s_end << " += " << fShapeInput[fAxes[i]] <<";\n";
            if (!fStepDims[i].isParam) {
               if (static_cast<IType>(fStepDims[i].dim) > 0 ) {
                  out << SP << "if (" << s_end << " < 0) " << s_end << " = 0;\n";
                  out << SP << "if (" << s_end << " > " << fShapeInput[fAxes[i]] << " ) " << s_end << " = " << fShapeInput[fAxes[i]] <<";\n";
               } else {
                  out << SP << "if (" << s_end << " < -1) " << s_end << " = -1;\n";
                  out << SP << "if (" << s_end << " > " << fShapeInput[fAxes[i]] << " - 1" << " ) " << s_end << " = " << fShapeInput[fAxes[i]] << " - 1;\n";
               }
            }
         }
         // special case if step is negative and shape are equal and step is negative
         else if (fEndDims[i].isParam && fEndDims[i].param == fShapeInput[fAxes[i]].param && !fStepDims[i].isParam && static_cast<IType>(fStepDims[i].dim) < 0 ) {
            fEnd[fAxes[i]] = Dim{ fEndDims[i].param + "-1" };
         }
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
