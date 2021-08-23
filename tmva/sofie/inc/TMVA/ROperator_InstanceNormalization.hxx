#ifndef TMVA_SOFIE_ROPERATOR_InstanceNormalization
#define TMVA_SOFIE_ROPERATOR_InstanceNormalization

#include "SOFIE_common.hxx"
#include "ROperator.hxx"
#include "RModel.hxx"


#include <cmath>
#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_InstanceNormalization final : public ROperator
{

private:

	/* Attributes */
	float fepsilon = 1e-05;

	std::string fNX;
	std::string fNScale;
	std::string fNB;
	std::string fNY;

	std::vector<size_t> fShapeX;
	std::vector<size_t> fShapeScale;
	std::vector<size_t> fShapeB;
	std::vector<size_t> fShapeY;
	
	std::string fType;

public:
	ROperator_InstanceNormalization() = delete;
	
	/* Constructor */
	ROperator_InstanceNormalization( float epsilon,
	std::string nameX, std::string nameScale, std::string nameB, 
	std::string nameY):
	fepsilon(epsilon),
	fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)), 
	fNB(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY))
	{
		if(std::is_same<T, float>::value){
			fType = "float";
		}
		else{
			throw
				std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a InstanceNormalization operator");
		}
	}
	

	std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) {
		ETensorType out = input[0];
		return {out};
	}

	std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) {
		if (input.size() != 3 ) {
			throw
				std::runtime_error("TMVA SOFIE InstanceNormalization Op Shape inference need 3 input tensors");
		}
		for(size_t i = 0; i < input.size(); i++) {
			if (input[i].size() != 4) {
				throw
				std::runtime_error("TMVA SOFIE InstanceNormalization Op Shape inference only accept tensor with 4 dimensions");
			}
		}

		auto ret = input; 
		return ret;
	}

	void Initialize(RModel& model){
		if (!model.CheckIfTensorAlreadyExist(fNX)) {
			throw
				std::runtime_error("TMVA SOFIE InstanceNormalization op Input Tensor " + fNX + " fnx is not found in model");
		}
		if (!model.CheckIfTensorAlreadyExist(fNScale)) {
			throw
				std::runtime_error("TMVA SOFIE InstanceNormalization op Input Tensor " + fNScale + " fns is not found in model");
		}
		if (!model.CheckIfTensorAlreadyExist(fNB)) {
			throw
				std::runtime_error("TMVA SOFIE InstanceNormalization op Input Tensor " + fNB + " fnb is not found in model");
		}

		fShapeX = model.GetTensorShape(fNX);
		if (fShapeX.size() != 4) {
			throw
				std::runtime_error("TMVA SOFIE InstanceNormalization Op input tensor " + fNX + " fnx is not of 4 dimensions");
		}
		
		fShapeScale = model.GetTensorShape(fNScale);
		fShapeB = model.GetTensorShape(fNB);
		fShapeY = fShapeX;
		model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
	}


	std::string Generate(std::string OpName){
		OpName = "op_" + OpName;
		if (fShapeX.empty()){
			throw std::runtime_error("TMVA SOFIE Instance Normalization called to Generate without being initialized first");
		}

		std::stringstream out;
		int length = 1;
		for(auto& i: fShapeX){
			length *= i;
		}
		//// Instance Normalization operator
		out << "\t" << "for (size_t n = 0; n < " << fShapeX[0] << "; n++) {\n";
		out << "\t" << "\t" << "for (size_t c = 0; c < " << fShapeX[1] << "; c++) {\n";

		//// calculate mean
		out << "\t" << "\t" << "\t" << "float "<< OpName<< "_mean = 0;\n";
		out << "\t" << "\t" << "\t" << "for (size_t h = 0; h < " << fShapeX[2] << "; h++) {\n";
		out << "\t" << "\t" << "\t" << "\t" << "for (size_t w = 0; w < " << fShapeX[3] << "; w++) {\n";
		out << "\t" << "\t" << "\t" << "\t" << "\t" << OpName<< "_mean += tensor_" << fNX << "[n * " << fShapeX[1] * fShapeX[2] * fShapeX[3] << " + c * "<< fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << " + w];\n";
		out << "\t" << "\t" << "\t" << "\t" << "}\n";
		out << "\t" << "\t" << "\t" << "}\n";
		out << "\t" << "\t" << "\t" << OpName<< "_mean = "<< OpName<< "_mean/"<<(fShapeX[2]*fShapeX[3])<<";\n";

		//// calculate var
		out << "\t" << "\t" << "\t" << "float "<< OpName<< "_var = 0;\n";
		out << "\t" << "\t" << "\t" << "for (size_t h = 0; h < " << fShapeX[2] << "; h++) {\n";
		out << "\t" << "\t" << "\t" << "\t" << "for (size_t w = 0; w < " << fShapeX[3] << "; w++) {\n";
		out << "\t" << "\t" << "\t" << "\t" << "\t" << OpName<< "_var += (tensor_" << fNX << "[n * " << fShapeX[1] * fShapeX[2] * fShapeX[3] << " + c * "<< fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << " + w] - "<< OpName<<"_mean) * (tensor_" << fNX << "[n * " << fShapeX[1] * fShapeX[2] * fShapeX[3] << " + c * "<< fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << " + w] - "<< OpName<<"_mean);\n";
		out << "\t" << "\t" << "\t" << "\t" << "}\n";
		out << "\t" << "\t" << "\t" << "}\n";
		out << "\t" << "\t" << "\t" << OpName<< "_var = "<< OpName<< "_var/"<<(fShapeX[2]*fShapeX[3])<<";\n";

		//// in op
		out << "\t" << "\t" << "\t" << "for (size_t h = 0; h < " << fShapeX[2] << "; h++) {\n";
		out << "\t" << "\t" << "\t" << "\t" << "for (size_t w = 0; w < " << fShapeX[3] << "; w++) {\n";
		out << "\t" << "\t" << "\t" << "\t" << "\t" << "tensor_" << fNY << "[n * " << fShapeX[1] * fShapeX[2] * fShapeX[3] << " + c * "<< fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << " + w] = ((tensor_" << fNX << "[n * " << fShapeX[1] * fShapeX[2] * fShapeX[3] << " + c * "<< fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << " + w] - " << OpName<<"_mean)/ std::sqrt(" << OpName<<"_var + "<<fepsilon<<" ))* " << "tensor_" << fNScale <<"[c] + " << "tensor_" <<fNB<<"[c];\n";
		out << "\t" << "\t" << "\t" << "\t" << "}\n";
		out << "\t" << "\t" << "\t" << "}\n";
		out << "\t" << "\t" << "}\n";	
		out << "\t" << "}\n";
		// std::cout<<out;
		return out.str();
	}

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_InstanceNormalization