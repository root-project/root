#include <memory>

#include "RModel.hxx"
#include "RModelParser_ONNX.hxx"

#include <cctype>
#include <algorithm>


using namespace TMVA::Experimental::SOFIE;

int main(){


   RModelParser_ONNX parser;
   RModel model = parser.Parse("./Linear_64.onnx");
   RModel model2 = std::move(model);
   model2.PrintRequiredInputTensors();
   model2.PrintInitializedTensors();
   model2.HeadInitializedTensors("18bias");
   model2.HeadInitializedTensors("0weight");

	std::cout << "===" << std::endl;

   model2.Generate();
   //model2.PrintGenerated();
   //model2.Initialize();
   model2.PrintInitializedTensors();
   model2.HeadInitializedTensors("6bias", 100);


	std::cout << "===" << std::endl;



   //model2.PrintGenerated();
   model2.OutputGenerated();
   //model2.PrintIntermediateTensors();
/*
	std::cout << "===" << std::endl;

	RModel model3;
	model3.AddInputTensorInfo("1", ETensorType::FLOAT, {1,2,3,4});
	//auto op = std::make_unique<ROperator_Transpose<float>>({3,2,1,0}, "1", "2");
	std::unique_ptr<ROperator>op ( new ROperator_Transpose<float>({3,2,1,0}, "1", "2")) ;
   model3.AddOperator(std::move(op));
	//op->Initialize(model3);
	//std::cout << (op->Generate("1"));

   model3.AddInputTensorInfo("3", ETensorType::FLOAT, {2,3});
   model3.AddInputTensorInfo("4", ETensorType::FLOAT, {3,2});
   std::unique_ptr<ROperator> op2 (new ROperator_Gemm<float> (1.0, 1.0, 0, 0, "3", "4", "5"));
   model3.AddOperator(std::move(op2));
   std::unique_ptr<ROperator> op3 (new ROperator_Relu<float> ("5", "6"));
   model3.AddOperator(std::move(op3));
   //op2->Initialize(model3);
   //std::cout << (op2->Generate("2"));

   model3.Generate();
	model3.PrintGenerated();
*/
}
