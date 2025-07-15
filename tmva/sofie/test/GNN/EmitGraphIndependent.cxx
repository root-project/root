// Author: Sanjiban Sengupta
// Description:
//    This program generates a RModel_GraphIndependent for testing

#include "TMVA/RModel_GraphIndependent.hxx"
#include "TMVA/FunctionList.hxx"
#include "TMVA/SOFIE_common.hxx"


int main()
{
    using namespace TMVA::Experimental::SOFIE;

    GraphIndependent_Init init;
    init.num_nodes=2;
    init.edges = {{1,0}};
    init.num_node_features = 2;
    init.num_edge_features = 2;
    init.num_global_features=2;
    init.filename = "Test_GraphIndependent";
    std::unique_ptr<RFunction_Update> func;
    func.reset(new RFunction_MLP(FunctionTarget::EDGES, 1, Activation::RELU, 1, GraphType::GraphIndependent));
    std::vector<std::string> kernel_tensors = {"graph_independent/edge_model/linear/w:0"};
    std::vector<std::string> bias_tensors = {"graph_independent/edge_model/linear/b:0"};
    std::vector<std::vector<std::string>> weight_tensors = {kernel_tensors,bias_tensors};
    func->AddInitializedTensors(weight_tensors);
    init.edges_update_block = std::move(func);
    func.reset(new RFunction_MLP(FunctionTarget::NODES, 1, Activation::RELU, 1, GraphType::GraphIndependent));
    kernel_tensors = {"graph_independent/node_model/linear/w:0"};
    bias_tensors = {"graph_independent/node_model/linear/b:0"};
    weight_tensors = {kernel_tensors,bias_tensors};
    func->AddInitializedTensors(weight_tensors);
    init.nodes_update_block = std::move(func);
    func.reset(new RFunction_MLP(FunctionTarget::GLOBALS, 1, Activation::RELU, 1, GraphType::GraphIndependent));
    kernel_tensors = {"graph_independent/global_model/linear/w:0"};
    bias_tensors = {"graph_independent/global_model/linear/b:0"};
    weight_tensors = {kernel_tensors,bias_tensors};
    func->AddInitializedTensors(weight_tensors);
    init.globals_update_block = std::move(func);

    float arr[] = { 0.41437718, -0.9389465 ,
                    -0.5776741 , -0.10654829};

    std::shared_ptr<void> data_0(malloc(4 * sizeof(float)), free);
    std::memcpy(data_0.get(), arr, 4 * sizeof(float));
    init.edges_update_block->GetFunctionBlock()->AddInitializedTensor("graph_independent/edge_model/linear/w:0",ETensorType::FLOAT,{2,2},data_0);
    float arr_b[2] = {0,0};
    std::shared_ptr<void> data_1(malloc(2 * sizeof(float)), free);
    std::memcpy(data_1.get(), arr_b, 2 * sizeof(float));
    init.edges_update_block->GetFunctionBlock()->AddInitializedTensor("graph_independent/edge_model/linear/b:0",ETensorType::FLOAT,{2},data_1);


    float arr_3[] = { 0.26933208,  0.16284496,
                    -0.23531358, -0.08090801};
    std::shared_ptr<void> data_6(malloc(4 * sizeof(float)), free);
    std::memcpy(data_6.get(), arr_3, 4* sizeof(float));
    init.globals_update_block->GetFunctionBlock()->AddInitializedTensor("graph_independent/global_model/linear/w:0",ETensorType::FLOAT,{2,2},data_6);
    std::shared_ptr<void> data_7(malloc(2 * sizeof(float)), free);
    std::memcpy(data_7.get(), arr_b, 2 * sizeof(float));
    init.globals_update_block->GetFunctionBlock()->AddInitializedTensor("graph_independent/global_model/linear/b:0",ETensorType::FLOAT,{2},data_7);

    float arr_4[] = {0.14061868, 0.30305472,
                    0.99493086, 0.36049494};
    std::shared_ptr<void> data_8(malloc(4 * sizeof(float)), free);
    std::memcpy(data_8.get(), arr_4, 4 * sizeof(float));
    init.nodes_update_block->GetFunctionBlock()->AddInitializedTensor("graph_independent/node_model/linear/w:0",ETensorType::FLOAT,{2,2},data_8);
    std::shared_ptr<void> data_9(malloc(2 * sizeof(float)), free);
    std::memcpy(data_9.get(), arr_b, 2 * sizeof(float));
    init.nodes_update_block->GetFunctionBlock()->AddInitializedTensor("graph_independent/node_model/linear/b:0",ETensorType::FLOAT,{2},data_9);

    RModel_GraphIndependent model(init);
    model.AddBlasRoutines({"Gemm"});
    model.Generate();
    model.OutputGenerated();
    
    func.reset();

   return 0;
}
