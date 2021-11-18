#ifndef TMVA_SOFIE_ROPERATOR_GEMM
#define TMVA_SOFIE_ROPERATOR_GEMM


#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <limits>

namespace TMVFA{
namespace Experimenatal{
namespace SOFIE{

    template <typename T>
    class ROperator_Concat final : public ROperator
    {
    private:
        std::vector<std::string> fInputs;
        std::string fOutput;
        int fAxis;
        std::vector<size_t>fShape;

    public:
        ROperator_Concat(){}
        ROperator_Concat(std::vector<std::string> inputs, int axis):
        fInputs(inputs), fAxis(axis){}

        std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
            return input;
        }

        std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
            std::vector<std::vector<size_t>> ret;
            std::vector<size_t> fOutputShape;
            int concat_dim=0;
            for(auto &it: inputs){
                concat_dim += it[fAxis];
            }
            if(fAxis==0){
                std::vector<size_t> fInputTensorShape = inputs[0];
                fOutputShape.assign(fInputTensorShape.begin()+1,fInputTensorShape.end());
                fOutputShape.insert(fOutputShape.begin(),concat_dim);
            }
            ret.push_back(fOutputShape);
            return ret;
        }

        void Initialize(RModel& model){
            std::vector<std::vector<<size_t>> fInputShapes;
            for(auto &it : fInputs){
                if (model.CheckIfTensorAlreadyExist(it) == false){
                throw std::runtime_error("TMVA SOFIE Relu Op Input Tensor is not found in model");
            }
            fInputShapes.push_back(model.GetTensorShape(it));
            }
            std::vector<size_t> fOutputShape = ShapeInference(fInputShapes)[0];
            model.AddIntermediateTensor(fOutput, model.GetTensorType(fInputs[0]), fOutputShape);
        }

        std::string Generate(std::string OpName){
            OpName = "op_"+OpName;
            const std::string SP = "   ";
            if(fShape.empty()){
                throw std::runtime_error("TMVA SOFIE Concat called to Generate without being initialized first");
            }
            std::stringstream out;
            out<<"\n//--------- Concat\n";
            if(fAxis==0){
                for(auto &it : fInputs){
                    out<<SP<<fOutput<<".insert("<<fOutput<<".end(),"<<it<<".begin(),"<<it<<".end());"
                }
            }

            return out.str();
        }


    };
}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_ROPERATOR_CONCAT
