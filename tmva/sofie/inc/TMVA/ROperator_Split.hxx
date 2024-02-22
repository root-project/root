#ifndef TMVA_SOFIE_ROPERATOR_Split
#define TMVA_SOFIE_ROPERATOR_Split

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class ROperator_Split final : public ROperator
{

private:

   std::string fNX;
   std::string fNS;
   std::vector<std::string> fNYs;
   std::vector<size_t> fInputShape;
   std::vector<std::vector<size_t>> fShapes;
   int fAxis = 0;


public:
   ROperator_Split(){}

   ROperator_Split(int axis, const std::string & nameX, const std::string & nameS, const std::vector<std::string> & namesY ):
      fNX(UTILITY::Clean_name(nameX)),
      fNS(UTILITY::Clean_name(nameS)),
      fAxis(axis)
   {
      for ( auto & name : namesY) {
         fNYs.push_back(UTILITY::Clean_name(name));
      }
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model){
       //input must be a graph input, or already initialized intermediate tensor0
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
         throw std::runtime_error("TMVA SOFIE Split Op Input Tensor is not found in model");
      }
      fInputShape = model.GetTensorShape(fNX);
      auto type = model.GetTensorType(fNX);

      int nouts = fNYs.size();
      // split in axis
      if (fAxis < 0) fAxis = fInputShape.size() + fAxis;
      if (fAxis < 0 || fAxis >= (int)  fInputShape.size())
        throw std::runtime_error("TMVA SOFIE Split Op has invalid axis : " + std::to_string(fAxis));

      // for the moment do not support split provided
      // this works if split values are all the same

      for (int i = 0; i < nouts; i++) {
         auto shape = fInputShape;
         // assume shape is computed from number of outputs
         // in reality if fNS is not null is obtained from input and it is in this case dynamic
         // but we need a shape to register for SOFIE
         shape[fAxis] = fInputShape[fAxis]/nouts;
         if (i == nouts -1 && shape[fAxis]*nouts != fInputShape[fAxis] )
            shape[fAxis] = fInputShape[fAxis] % nouts;

         model.AddIntermediateTensor(fNYs[i], type, shape);
         fShapes.push_back(shape);
      }
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapes.empty()) {
         throw std::runtime_error("TMVA SOFIE Split called to Generate without being initialized first");
      }
      std::stringstream out;
      int nouts = fNYs.size();
      auto inputStride = TMVA::Experimental::SOFIE::UTILITY::ComputeStrideFromShape(fInputShape);
      out << "\n//------ Split\n";
      for (int i = 0; i < nouts; i++) {
          // we support now split in equal part except last one
         // if a tensor with split shape is provided check if consistent on what we expect
         if (!fNS.empty()) {
            std::string splitValue = "tensor_" + fNS + "[" + std::to_string(i) + "]";
            out << SP << "if (" << splitValue << " != " << fShapes[i][fAxis] << ")\n";
            out << SP << SP << "throw std::runtime_error(\"TMVA SOFIE Split operator has un-supported split input \" + std::to_string("
               << splitValue << ") );\n";
         }
         // case split axis is first split is on block of consecutive memory
         auto l = TMVA::Experimental::SOFIE::ConvertShapeToLength(fShapes[i]);
         auto outStride = TMVA::Experimental::SOFIE::UTILITY::ComputeStrideFromShape(fShapes[i]);
         if (fAxis == 0) {
            out << SP << "std::copy(tensor_" << fNX << " + " << i*l << ", tensor_" << fNX << " + " << (i+1)*l
               << ", tensor_" << fNYs[i] << ");\n";
         }
         else {
            // here is more complicated:
            // use formula to express global index as function of coordinate indices:
            //   id = s[0]*i0 + s[1]*i1+ s[2]*i2,....  where s[] is the stride
            // and the inverse :
            // i0 = id/s[0];  i1 = (id % s[0])/s[1]   i2 = (id % s[1])/s[2] .....
            // for the split axis the index in that axis for the input tensor is:
            //   i_axis = (id % s[axis-1])/s[axis] + split_number * axis_split_size
            // we need then to compute the i0,i1,.. iN and then correct the one where teh split is happening
            out << SP << "for (int id = 0; id < " << l << "; id++) {\n";
            // compute the coordinate indices
            int ndim = fShapes[i].size();
            for (int j = 0; j < ndim; j++ ) {
               out << SP << SP << "size_t inputId = ";
               if (j == 0) {
                  out << inputStride[0] << " * (id/" << outStride[0] << ");\n";
               } else {
                  // case fAxis = 0 has been already dealt before
                  out << "inputId += " << inputStride[j] << " * ( (id % " << outStride[j-1] << ")/" << outStride[j];
                  // we should implement here the case where an input shape is provided
                  if (j == fAxis) out << " + " << i << " * " << int(fInputShape[j]/nouts);
                  out << ");\n";
               }
               out << SP << SP << "tensor_" << fNYs[i] << " [id] =  " << " tensor_" << fNX << "[inputId];\n";
            }
            out << SP << "}\n";
         }
      }

      out << SP << "\n";

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Split
