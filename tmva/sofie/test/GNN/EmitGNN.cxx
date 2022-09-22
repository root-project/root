// Author: Sanjiban Sengupta
// Description:
//    This program generates a RModel_GNN for testing

#include <iostream>

#include "TMVA/RModel_GNN.hxx"
#include "TMVA/FunctionList.hxx"
#include "TMVA/SOFIE_common.hxx"


using namespace TMVA::Experimental::SOFIE;

int main(){

    GNN_Init init;
    init.num_nodes=2;
    init.edges = {{1,0}};
    init.num_node_features = 2;
    init.num_edge_features = 2;
    init.num_global_features=2;
    init.filename = "Test_Graph";
    std::unique_ptr<RFunction_Update> func;
    func.reset(new RFunction_MLP(FunctionTarget::EDGES, 1, 0));
    std::vector<std::string> kernel_tensors = {"graph_network/edge_block/linear/w:0"};
    std::vector<std::string> bias_tensors = {"graph_network/edge_block/linear/b:0"};
    std::vector<std::vector<std::string>> weight_tensors = {kernel_tensors,bias_tensors};
    func->AddInitializedTensors(weight_tensors);
    init.edges_update_block = std::move(func);
    func.reset(new RFunction_MLP(FunctionTarget::NODES, 1, 0));
    kernel_tensors = {"graph_network/node_block/linear/w:0"};
    bias_tensors = {"graph_network/node_block/linear/b:0"};
    weight_tensors = {kernel_tensors,bias_tensors};
    func->AddInitializedTensors(weight_tensors);
    init.nodes_update_block = std::move(func);
    func.reset(new RFunction_MLP(FunctionTarget::GLOBALS, 1, 0));
    kernel_tensors = {"graph_network/global_block/linear/w:0"};
    bias_tensors = {"graph_network/global_block/linear/b:0"};
    weight_tensors = {kernel_tensors,bias_tensors};
    func->AddInitializedTensors(weight_tensors);
    init.globals_update_block = std::move(func);
    std::unique_ptr<RFunction_Aggregate> func_agg;
    func_agg.reset(new RFunction_Sum());
    init.node_global_agg_block = std::move(func_agg);
    func_agg.reset(new RFunction_Sum());
    init.edge_global_agg_block = std::move(func_agg);
    func_agg.reset(new RFunction_Sum());
    init.edge_node_agg_block = std::move(func_agg);
    float arr[] = {-0.11413302,  0.49974972,
        -0.15535775,  0.06823446,
        -0.23475496, -0.38286394,
            0.16671045,  0.1850846 ,
        -0.4561586 ,  0.3438921 ,
            0.10795765,  0.49663377,
        -0.6825379 ,  0.25719026,
            0.24045151,  0.13871197};

    std::shared_ptr<void> data_0(malloc(16 * sizeof(float)), free);
    std::memcpy(data_0.get(), arr, 16 * sizeof(float));
    init.edges_update_block->GetFunctionBlock()->AddInitializedTensor("graph_network/edge_block/linear/w:0",ETensorType::FLOAT,{8,2},data_0);
    float arr_b[2] = {0,0};
    std::shared_ptr<void> data_1(malloc(2 * sizeof(float)), free);
    std::memcpy(data_1.get(), arr_b, 2 * sizeof(float));
    init.edges_update_block->GetFunctionBlock()->AddInitializedTensor("graph_network/edge_block/linear/b:0",ETensorType::FLOAT,{2},data_1);


    float arr_3[] = {-0.2939381 ,  0.29374102,
            0.39915594,  0.22476648,
            0.07345466, -0.14384857,
            0.15938309,  0.19942378,
        -0.14101209, -0.57209873,
        -0.26913098,  0.5071538 };
    std::shared_ptr<void> data_6(malloc(12 * sizeof(float)), free);
    std::memcpy(data_6.get(), arr_3, 12* sizeof(float));
    init.globals_update_block->GetFunctionBlock()->AddInitializedTensor("graph_network/global_block/linear/w:0",ETensorType::FLOAT,{6,2},data_6);
    std::shared_ptr<void> data_7(malloc(2 * sizeof(float)), free);
    std::memcpy(data_7.get(), arr_b, 2 * sizeof(float));
    init.globals_update_block->GetFunctionBlock()->AddInitializedTensor("graph_network/global_block/linear/b:0",ETensorType::FLOAT,{2},data_7);

    float arr_4[] = {-0.49379408,  0.64651597,
        -0.48153368,  0.04505178,
            0.36477658,  0.40089235,
        -0.26732066, -0.40632117,
            0.61156213,  0.563861  ,
            0.28957435,  0.46537283};
    std::shared_ptr<void> data_8(malloc(12 * sizeof(float)), free);
    std::memcpy(data_8.get(), arr_4, 12 * sizeof(float));
    init.nodes_update_block->GetFunctionBlock()->AddInitializedTensor("graph_network/node_block/linear/w:0",ETensorType::FLOAT,{6,2},data_8);
    std::shared_ptr<void> data_9(malloc(2 * sizeof(float)), free);
    std::memcpy(data_9.get(), arr_b, 2 * sizeof(float));
    init.nodes_update_block->GetFunctionBlock()->AddInitializedTensor("graph_network/node_block/linear/b:0",ETensorType::FLOAT,{2},data_9);

    RModel_GNN model(init);
    model.AddBlasRoutines({"Gemm"});
    model.Generate();
    model.OutputGenerated();

    func.reset();
    func_agg.reset();

    return 0;
}
