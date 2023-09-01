/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides a simple example for the parsing of PyTorch .pt file
/// into RModel object and further generating the .hxx header files for inference.
///
/// \macro_code
/// \macro_output
/// \author Sanjiban Sengupta

using namespace TMVA::Experimental;

TString pythonSrc = "\
import torch\n\
import torch.nn as nn\n\
\n\
model = nn.Sequential(\n\
           nn.Linear(32,16),\n\
           nn.ReLU(),\n\
           nn.Linear(16,8),\n\
           nn.ReLU()\n\
           )\n\
\n\
criterion = nn.MSELoss()\n\
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)\n\
\n\
x=torch.randn(2,32)\n\
y=torch.randn(2,8)\n\
\n\
for i in range(500):\n\
    y_pred = model(x)\n\
    loss = criterion(y_pred,y)\n\
    optimizer.zero_grad()\n\
    loss.backward()\n\
    optimizer.step()\n\
\n\
model.eval()\n\
m = torch.jit.script(model)\n\
torch.jit.save(m,'PyTorchModel.pt')\n";


void TMVA_SOFIE_PyTorch(){

    //Running the Python script to generate PyTorch .pt file
    TMVA::PyMethodBase::PyInitialize();

    TMacro m;
    m.AddLine(pythonSrc);
    m.SaveSource("make_pytorch_model.py");
    gSystem->Exec(TMVA::Python_Executable() + " make_pytorch_model.py");

    //Parsing a PyTorch model requires the shape and data-type of input tensor
    //Data-type of input tensor defaults to Float if not specified
    std::vector<size_t> inputTensorShapeSequential{2,32};
    std::vector<std::vector<size_t>> inputShapesSequential{inputTensorShapeSequential};

    //Parsing the saved PyTorch .pt file into RModel object
    SOFIE::RModel model = SOFIE::PyTorch::Parse("PyTorchModel.pt",inputShapesSequential);

    //Generating inference code
    model.Generate();
    model.OutputGenerated("PyTorchModel.hxx");

    //Printing required input tensors
    std::cout<<"\n\n";
    model.PrintRequiredInputTensors();

    //Printing initialized tensors (weights)
    std::cout<<"\n\n";
    model.PrintInitializedTensors();

    //Printing intermediate tensors
    std::cout<<"\n\n";
    model.PrintIntermediateTensors();

    //Checking if tensor already exist in model
    std::cout<<"\n\nTensor \"0weight\" already exist: "<<std::boolalpha<<model.CheckIfTensorAlreadyExist("0weight")<<"\n\n";
    std::vector<size_t> tensorShape = model.GetTensorShape("0weight");
    std::cout<<"Shape of tensor \"0weight\": ";
    for(auto& it:tensorShape){
        std::cout<<it<<",";
    }
    std::cout<<"\n\nData type of tensor \"0weight\": ";
    SOFIE::ETensorType tensorType = model.GetTensorType("0weight");
    std::cout<<SOFIE::ConvertTypeToString(tensorType);

    //Printing generated inference code
    std::cout<<"\n\n";
    model.PrintGenerated();
}
