#ifndef TMVA_SOFIE_ROPERATOR_Clip
#define TMVA_SOFIE_ROPERATOR_Clip

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <numeric>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Clip final : public ROperator
{

private:
    std::string fNMin;
    std::string fNMax;
    std::string fNX;
    std::string fNY;
    ETensorType fType;
    std::vector<size_t> fShape;

public:
    ROperator_Clip(){}
    ROperator_Clip(std::string nameMin, std::string nameMax, std::string nameX, std::string nameY):
        fNMin(UTILITY::Clean_name(nameMin)), fNMax(UTILITY::Clean_name(nameMax)), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){}

    std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
        return input;
    }

    std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
        auto ret = input; //suggest copy to compiler
        return ret;
    }

    void Initialize(RModel& model){
        //input must be a graph input, or already initialized intermediate tensor
        if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("TMVA SOFIE Clip Op Input Tensor is not found in model");
        }
        fShape = model.GetTensorShape(fNX);
        fType = model.GetTensorType(fNX);
        model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
    }


    std::string Generate(std::string OpName){
        OpName = "op_" + OpName;
        if (fShape.empty()) {
            throw std::runtime_error("TMVA SOFIE Clip operator called to Generate without being initialized first");
        }
        std::stringstream out;
        size_t length = ConvertShapeToLength(fShape);
        out << "\n//------ Clip\n";
        out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
        if(fNMin.length()) {
            out << SP << SP << "if(tensor_" << fNX << "[id] < tensor_" << fNMin << "[0])\n";
            out << SP << SP << SP << "tensor_" << fNY << "[id] = tensor_" << fNMin << "[0];\n";
        }
        else {
            out << SP << SP << "if(tensor_" << fNX << "[id] < std::numeric_limits<" << ConvertTypeToString(fType) << ">::lowest())\n";
            out << SP << SP << SP << "tensor_" << fNY << "[id] = std::numeric_limits<" << ConvertTypeToString(fType) << ">::lowest();\n";
        }
        if(fNMax.length()) {
            out << SP << SP << "else if(tensor_" << fNX << "[id] > tensor_" << fNMax << "[0])\n";
            out << SP << SP << SP << "tensor_" << fNY << "[id] = tensor_" << fNMax << "[0];\n";
        }
        else {
            out << SP << SP << "else if(tensor_" << fNX << "[id] > std::numeric_limits<" << ConvertTypeToString(fType) << ">::max())\n";
            out << SP << SP << SP << "tensor_" << fNY << "[id] = std::numeric_limits<" << ConvertTypeToString(fType) << ">::max();\n";
        }
        out << SP << SP << "else\n";
        out << SP << SP << SP << "tensor_" << fNY << "[id] = tensor_" << fNX << "[id];\n";
        out << SP << "}\n";
        return out.str();
    }

    std::vector<std::string> GetStdLibs() { return { std::string("cmath") };}
    };

    }//SOFIE
    }//Experimental
    }//TMVA


    #endif //TMVA_SOFIE_ROPERATOR_Clip
