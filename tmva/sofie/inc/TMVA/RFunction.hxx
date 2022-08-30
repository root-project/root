#ifndef TMVA_SOFIE_RFUNCTION
#define TMVA_SOFIE_RFUNCTION

#include <any>
#include "TMVA/RModel_GNN.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RModel;

enum class FunctionType{
        UPDATE=0, AGGREGATE=1
};
enum class FunctionTarget{
        INVALID=0, NODES=1, EDGES=2, GLOBALS=3
};
enum class FunctionRelation{
        INVALID=0, NODES_GLOBALS=1, EDGES_GLOBALS=2, EDGES_NODES=3
};
class RFunction{
    protected:
        std::string fFuncName;
        FunctionType fType;
        std::shared_ptr<RModel> function_block;
    public:
        RFunction(){}
        virtual ~RFunction(){}
        FunctionType GetFunctionType(){
                return fType;
        }
        std::shared_ptr<RModel> GetFunctionBlock(){
                return function_block;
        }

        RFunction(std::string funcName, FunctionType type):
                fFuncName( UTILITY::Clean_name(funcName)),fType(type){}
        
        std::string GenerateModel(std::string filename, long read_pos=0){
            function_block->SetFilename(filename);
            function_block->Generate(Options::kGNNComponent,1,read_pos);
            std::string modelGenerationString;
            if(fType == FunctionType::UPDATE)
                modelGenerationString = "\n//--------- GNN_Update_Function---"+fFuncName+"\n"+function_block->ReturnGenerated();
            else        
                modelGenerationString = "\n//--------- GNN_Aggregate_Function---"+fFuncName+"\n"+function_block->ReturnGenerated();

            return modelGenerationString;
        }

        std::string Generate(std::vector<std::string> inputPtrs){
            std::string inferFunc = fFuncName+"::infer(";
            for(auto&it : inputPtrs){
                inferFunc+=it;
                inferFunc+=",";
            }
            inferFunc.pop_back();
            inferFunc+=");";
            return inferFunc;
        }

};

class RFunction_Update: public RFunction{
        protected:
                FunctionTarget fTarget;
                std::vector<std::string> fInputTensors;
        public:
        virtual ~RFunction_Update(){}
        RFunction_Update(){}
                RFunction_Update(FunctionTarget target): fTarget(target){
                        switch(target){
                                case FunctionTarget::EDGES:{
                                        fFuncName = "Edge_Update";
                                        break;
                                } 
                                case FunctionTarget::NODES: {
                                        fFuncName = "Node_Update";
                                        break;
                                }
                                case FunctionTarget::GLOBALS: {
                                        fFuncName = "Global_Update";
                                        break;
                                }
                                default:
                                        throw std::runtime_error("Invalid target for Update function");
                        }
                        fType = FunctionType::UPDATE;
                        function_block = std::make_unique<RModel>(fFuncName);
         
                }
                virtual void AddInitializedTensors(std::any){};
                virtual void Initialize(){};
                void AddInputTensors(std::vector<std::vector<std::size_t>> fInputShape){
                        for(long unsigned int i=0; i<fInputShape.size(); ++i){
                                function_block->AddInputTensorInfo(fInputTensors[i],ETensorType::FLOAT, fInputShape[i]);
                                function_block->AddInputTensorName(fInputTensors[i]);
                        }
                }
};

class RFunction_Aggregate: public RFunction{
        protected:
                FunctionRelation fRelation;
                std::vector<std::vector<std::string>> fInputTensors;
        public:
        virtual ~RFunction_Aggregate(){}
        RFunction_Aggregate(){}
                RFunction_Aggregate(FunctionRelation relation):fRelation(relation){
                        switch (relation)
                        {
                                case FunctionRelation::NODES_GLOBALS:{
                                        fFuncName = "Nodes_Global_Aggregate";
                                        break;
                                }
                                case FunctionRelation::EDGES_GLOBALS:{
                                        fFuncName = "Edges_Global_Aggregate";
                                        break;
                                }
                                case FunctionRelation::EDGES_NODES:{
                                        fFuncName = "Edges_Nodes_Aggregate";
                                        break;
                                }
                                default:
                                        throw std::runtime_error("Invalid relation for Aggregate function");

                        }
                        fType = FunctionType::AGGREGATE;
                        function_block = std::make_unique<RModel>(fFuncName);
                }
                virtual void AddInitializedTensors(std::any){};
                virtual void Initialize(){};
                void AddInputTensors(std::vector<std::vector<std::vector<std::size_t>>> fInputShape){
                        for(long unsigned int i=0; i<fInputShape.size(); ++i){
                                        for(long unsigned int j=0;j<fInputShape[0].size();++j){
                                                function_block->AddInputTensorInfo(fInputTensors[i][j],ETensorType::FLOAT, fInputShape[i][j]);
                                                function_block->AddInputTensorName(fInputTensors[i][j]);
                                        }
                                }
                }
};


}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_RFUNCTION