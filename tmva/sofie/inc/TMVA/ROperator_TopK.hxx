#ifndef TMVA_SOFIE_ROPERATOR_TOPK
#define TMVA_SOFIE_ROPERATOR_TOPK

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_TopK final : public ROperator {

private:
   int fAttrAxis;
   int fAttrLargest;
   int fAttrSorted;

   size_t k;
   std::string fNK;
   std::string fNX;
   std::string fNVal;
   std::string fNInd;
   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeY;
   std::string fType;

public:
   ROperator_TopK() {}
   ROperator_TopK(int attr_axis, int attr_largest, int attr_sorted, std::string nameK, std::string nameX, std::string nameVal, std::string nameInd)
      : fAttrAxis(attr_axis),
        fAttrLargest(attr_largest),
        fAttrSorted(attr_sorted),
        fNK(UTILITY::Clean_name(nameK)),
        fNX(UTILITY::Clean_name(nameX)),
        fNVal(UTILITY::Clean_name(nameVal)),
        fNInd(UTILITY::Clean_name(nameInd)){}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) {
         ETensorType ret = input[0];
         return {ret, ret};
      }

   std::vector<std::vector<size_t>> ShapeInference(const std::vector<std::vector<size_t>> input) {
      if (input.size() != 2) {
         throw std::runtime_error("TMVA SOFIE TopK Op Shape Inference needs exactly 2 input tensors");
      }

      auto shape = input[0]; // Shape format: [ m x n x o x p ... ]
      if (fAttrAxis < 0) {
         fAttrAxis += shape.size();
      }
      if (fAttrAxis < 0 || fAttrAxis >= shape.size()) {
         throw std::runtime_error("TMVA SOFIE TopK Op axis value is out of bounds");
      }

      // set the dimension at the specified axis to k
      shape[fAttrAxis] = k; // Modified shape: [ m x n x k x p ... ]
      return {shape, shape};
   }


   void Initialize(RModel &model)
   {
      if (model.CheckIfTensorAlreadyExist(fNX) == false) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE TopK Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNK) == false) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE TopK Op Input Tensor i.e. K is not found in model");
      }

      fShapeX=model.GetTensorShape(fNX);
      auto fShapeK=model.GetTensorShape(fNK);
      auto kptr=static_cast<int64_t *>(model.GetInitializedTensorData(fNK).get());
      k=*kptr;
      fAttrAxis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      if(fAttrAxis>=fShapeX.size()){
         throw
            std::runtime_error("TMVA::SOFIE ONNX TopK op axis = "+ std::to_string(fAttrAxis) +" value exeeds size of tensor " +fNX+" of size "+fShapeX.size()+" .");
      }
      if(k>fShapeX[fAttrAxis]){
         throw
            std::runtime_error("TMVA::SOFIE ONNX TopK op k = "+ std::to_string(k) +" value exeeds value of tensor " +fNX+" of size "+fShapeX.size()+" at axis= "+std::to_string(fAttrAxis)+".");
      }
      // fShapeX = model.GetTensorShape(fNX); //  [ m x n x o x p ... ]
      // if(k[0]>=fShapeX.size()){
      //    throw
      //       std::runtime_error("TMVA::SOFIE ONNX TopK op k = "+ std::to_string(k[0]) +"value exeeds size of tensor " +fNX+" of size "+fShapeX.size()+" .");
      // }
      // fShapeY.push_back(2);
      // for (auto i : fShapeX)
      //    fShapeY.push_back(i); //  [ 2 x m x n x o x p ... ]
      // size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      // fShapeY[axis] = k[0]; //  [ 2 x m x n x K x p ... ]
      fShapeY=ShapeInference({fShapeX,fShapeK})[0];
      // for(int i=0;i<fShapeX.size();i++)
      // std::cout<<fShapeX[i]<<" ";
      // std::cout<<"\ny size -> "<<fShapeY.size()<<std::endl;


      model.AddIntermediateTensor(fNVal, model.GetTensorType(fNX), fShapeY);
      model.AddIntermediateTensor(fNInd, model.GetTensorType(fNX), fShapeY);
      fType = ConvertTypeToString(model.GetTensorType(fNX));
   }

   std::string Generate(std::string OpName)
   {
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator TopK called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t size = fShapeX.size();
      size_t axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;
      out << "\n" << SP << "//------ TopK\n";

      size_t length=ConvertShapeToLength(fShapeX);
      size_t bound=1;
      for(int i=0;i<axis;i++)bound*=fShapeX[i]; //bound decider

      // first we create boundaries in the input
      // [m,n,o,k,p] => boundary's size = length/(m*n*o)
      size_t groupSize = length/bound; //final search space for topK elements

      size_t jump= groupSize/fShapeX[fAttrAxis];
      //candidates to check in group
      size_t prod =1;
      for(int i=fAttrAxis+1;i<size;i++){
         prod*=fShapeX[i];
      }
      size_t numOfChecksInGrp=groupSize/jump;
      size_t numOfCheckersInGrp=groupSize/numOfChecksInGrp;

      // for(int i=0;i<length;i++){
      //    if(i==groupSize)dim=0;
      // }
      out<<SP<<"size_t itr = 0, p=0;\n";
      out<<SP<<"std::vector<std::vector<std::pair<float,int>>>groupElements;\n";
      out<<SP<<"for(long i=0;i<"<<length<<";i++) {\n";
      //main logic
      out<<SP<<SP<<"size_t tempitr=0, j=0;\n";
      out<<SP<<SP<<"std::vector<std::pair<float,int>>elements;\n";
      out<<SP<<SP<<"while(tempitr < "<<groupSize<<"){\n         elements.push_back({tensor_"<<fNX<<"[i+tempitr]"<<",tempitr});\n"<<SP<<SP<<SP<<"j++;\n"<<"         tempitr = j*"<<jump<<";\n"<<"\n"<<"      }\n";
      if (fAttrSorted) {
         if (fAttrLargest) {
            out<<SP<<SP << "std::sort(elements.begin(),elements.end(),[](std::pair<float,int>a,std::pair<float,int>b){return "
                   "a.first>b.first;});\n";
         } else
            out<<SP<<SP << "std::sort(elements.begin(),elements.end(),[](std::pair<float,int>a,std::pair<float,int>b){return "
                   "a.first<b.first;});\n";
      } else
         out<<SP<<SP << "std::sort(elements.begin(),elements.end());\n";

      out<<SP<<SP<<"itr++;\n";
      out<<SP<<SP<<"std::vector<std::pair<float,int>>kelems;\n";
      out<<SP<<SP<<"for(int j=0; j < "<<k<<";j++){\n         kelems.push_back({elements[j].first,elements[j].second});\n"<<SP<<SP<<"}\n";
      out<<SP<<SP<<"groupElements.push_back(kelems);\n";
      out<<SP<<SP<<"if(itr == "<<numOfCheckersInGrp<<"){\n         itr = 0;\n         i += "<<groupSize-numOfCheckersInGrp/*to compensate the default i++*/<<";\n";
         out<<SP<<SP<<SP<<"for(int j=0; j < groupElements[0].size();j++) {\n";
            out<<SP<<SP<<SP<<SP<<"for(int k=0; k < groupElements.size();k++) {\n";
            out<<SP<<SP<<SP<<SP<<SP<<"tensor_"<<fNVal<<"[p]=(groupElements[k][j].first);\n";
            out<<SP<<SP<<SP<<SP<<SP<<"tensor_"<<fNInd<<"[p++]=(groupElements[k][j].second);\n";
            out<<SP<<SP<<SP<<SP<<"}\n";// end for
         out<<SP<<SP<<SP<<"}\n";// end for
         out<<SP<<SP<<SP<<"groupElements.clear();\n";
      out<<SP<<SP<<"}\n";//end if
      out<<SP<<"\n}\n"; // end for

      return out.str();
   }
};

} // nameSPace SOFIE
} // nameSPace Experimental
} // nameSPace TMVA

#endif // TMVA_SOFIE_ROPERATOR_TOPK