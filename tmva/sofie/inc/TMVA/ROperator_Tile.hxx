#ifndef TMVA_SOFIE_ROPERATOR_TILE
#define TMVA_SOFIE_ROPERATOR_TILE

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <cassert>
#include <sstream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_Tile final : public ROperator {

private:
    std::string fNData;        // input data tensor name
    std::string fNReps;        // repetitions tensor name
    std::string fNOutput;      // output tensor name
    std::vector<size_t> fShapeInput; // input shape data
    std::vector<size_t> fShapeOutput; // output shape data

public:
    ROperator_Tile() {}
    ROperator_Tile(std::string nameData, std::string nameReps, std::string nameOutput)
        : fNData(UTILITY::Clean_name(nameData)), fNReps(UTILITY::Clean_name(nameReps)),
          fNOutput(UTILITY::Clean_name(nameOutput)) {}

    // output type is same as input
    std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) {
        auto ret = std::vector<ETensorType>(1, input[0]);
        return ret;
    }

    // output shape
    std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) {
        std::vector<std::vector<size_t>> ret;
        auto &input_shape = input[0];

        if (input.size() != 2) throw std::runtime_error("TMVA SOFIE Tile Op needs 2 input tensors");

        auto &reps_shape = input[1];
        if (reps_shape.size() != 1) throw std::runtime_error("TMVA SOFIE Tile Op's repetitions tensor must be 1-dimensional");

        size_t reps = reps_shape[0];
        auto output_shape = input_shape;
        for (size_t i = 0; i < input_shape.size(); ++i) {
            output_shape[i] *= reps;
        }

        ret.push_back(output_shape);
        return ret;
    }

    void Initialize(RModel &model) {
        if (model.CheckIfTensorAlreadyExist(fNData) == false) {
            // input must be a graph input, or already initialized intermediate tensor
            throw std::runtime_error("TMVA Tile Op Input Tensor " + fNData + "  is not found in model");
        }
        fShapeInput = model.GetTensorShape(fNData);
        if (model.CheckIfTensorAlreadyExist(fNReps) == false) {
            // input must be a graph input, or already initialized intermediate tensor
            throw std::runtime_error("TMVA Tile Op Repetitions Tensor " + fNReps + "  is not found in model");
        }
        auto reps_shape = model.GetTensorShape(fNReps);
        if (reps_shape.size() != 1) throw std::runtime_error("TMVA SOFIE Tile Op's repetitions tensor must be 1-dimensional");
        if (reps_shape[0] != 1) throw std::runtime_error("TMVA SOFIE Tile Op's repetitions tensor must contain a single value");
        auto reps_data = model.GetInitializedTensorData<int>(fNReps);
        int reps = reps_data[0];
        
        fShapeOutput = ShapeInference({fShapeInput, {static_cast<size_t>(reps)}})[0];
        model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNData), fShapeOutput);
    }

    std::string Generate(std::string OpName) {
        OpName = "op_" + OpName;
        if (fShapeInput.empty() || fShapeOutput.empty()) {
            throw std::runtime_error("TMVA SOFIE Tile Op called to Generate without being initialized first");
        }

        size_t input_length = ConvertShapeToLength(fShapeInput);
        size_t output_length = ConvertShapeToLength(fShapeOutput);
        if (output_length != input_length) {
            throw std::runtime_error("TMVA SOFIE Tile Op : wrong output shape - is " +
                                    ConvertShapeToString(fShapeOutput) + " and input is " +
                                    ConvertShapeToString(fShapeInput));
        }

        std::stringstream out;
        out << SP << "///-------- Tile operator\n" << std::endl;
        out << SP << "size_t in_offset = 0;" << std::endl;
        out << SP << "size_t out_offset = 0;" << std::endl;
        out << SP << "size_t reps = " << fNReps << "[0];" << std::endl;
        out << SP << "for (size_t i = 0; i < reps; ++i) {" << std::endl;
        out << SP << "    std::copy(tensor_" << fNData << " + in_offset, tensor_" << fNData << " + in_offset + " << input_length << ", tensor_" << fNOutput << " + out_offset);" << std::endl;
        out << SP << "    out_offset += " << input_length << ";" << std::endl;
        out << SP << "}" << std::endl;
        return out.str();
    }
};

} // SOFIE
} // Experimental
} // TMVA

#endif // TMVA_SOFIE_ROPERATOR_TILE
