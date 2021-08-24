#ifndef TMVA_SOFIE_ROPERATOR_RNN
#define TMVA_SOFIE_ROPERATOR_RNN

#include "TMVA/RModel.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/SOFIE_common.hxx"

#include <memory>
#include <sstream>
#include <vector>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T> class ROperator_RNN final : public ROperator {
 private:
   std::vector<float> fAttrActivationAlpha;
   std::vector<float> fAttrActivationBeta;
   std::vector<std::string> fAttrActivations;
   float fAttrClip;
   std::string fAttrDirection;
   size_t fAttrHiddenSize;
   size_t fAttrLayout;

   std::string fNX;
   std::string fNW;
   std::string fNR;
   std::string fNB;
   std::string fNSequence_lens;
   std::string fNInitial_h;
   std::string fNY;
   std::string fNY_h;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeW;
   std::vector<size_t> fShapeR;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeSequence_lens;
   std::vector<size_t> fShapeInitial_h;
   std::vector<size_t> fShapeY;
   std::vector<size_t> fShapeY_h;

   std::string fType;

 public:
   ROperator_RNN() {}

   ROperator_RNN(std::vector<float> activation_alpha,
                 std::vector<float> activation_beta,
                 std::vector<std::string> activations, float clip,
                 std::string direction, size_t hidden_size, size_t layout,
                 std::string nameX, std::string nameW, std::string nameR,
                 std::string nameB, std::string nameSequence_lens,
                 std::string nameInitial_h, std::string nameY,
                 std::string nameY_h)
       : fAttrActivationAlpha(activation_alpha),
         fAttrActivationBeta(activation_beta), fAttrActivations(activations),
         fAttrClip(clip), fAttrDirection(direction),
         fAttrHiddenSize(hidden_size), fAttrLayout(layout),
         fNX(UTILITY::Clean_name(nameX)), fNW(UTILITY::Clean_name(nameW)),
         fNR(UTILITY::Clean_name(nameR)), fNB(UTILITY::Clean_name(nameB)),
         fNSequence_lens(UTILITY::Clean_name(nameSequence_lens)),
         fNInitial_h(UTILITY::Clean_name(nameInitial_h)),
         fNY(UTILITY::Clean_name(nameY)), fNY_h(UTILITY::Clean_name(nameY_h)) {
      if (std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw std::runtime_error(
             "TMVA SOFIE Encountered unsupported type parsing a RNN operator");
      }
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) {
      ETensorType out = input[0];
      return {out, out};
   }

   std::vector<std::vector<size_t>>
   ShapeInference(std::vector<std::vector<size_t>> input) {
      size_t num_directions = input[1][0];
      size_t hidden_size = input[1][1];
      if (fAttrLayout == 0) {
         size_t seq_length = input[0][0];
         size_t batch_size = input[0][1];
         std::vector<std::vector<size_t>> ret(
             {{seq_length, num_directions, batch_size, hidden_size},
              {num_directions, batch_size, hidden_size}});
         return ret;
      } else {
         size_t batch_size = input[0][0];
         size_t seq_length = input[0][1];
         std::vector<std::vector<size_t>> ret(
             {{batch_size, seq_length, num_directions, hidden_size},
              {batch_size, num_directions, hidden_size}});
         return ret;
      }
   }

   void Initialize(RModel &model) {
      // Check the input and output tensors
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNX + "  is not found in model.");
      }
      fShapeX = model.GetTensorShape(fNX);
      if (fShapeX.size() != 3) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNX + " is not of 3 dimensions.");
      }
      if (!model.CheckIfTensorAlreadyExist(fNW)) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNW + "  is not found in model.");
      }
      fShapeW = model.GetTensorShape(fNW);
      if (fShapeW.size() != 3) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNW + " is not of 3 dimensions.");
      }
      if (!model.CheckIfTensorAlreadyExist(fNR)) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNR + "  is not found in model.");
      }
      fShapeR = model.GetTensorShape(fNR);
      if (fShapeR.size() != 3) {
         throw std::runtime_error("TMVA SOFIE RNN Op input tensor " + fNR + " is not of 3 dimensions.");
      }
      if (!fNB.empty()) {
         if (!model.CheckIfTensorAlreadyExist(fNB)) {
            throw std::runtime_error("TMVA SOFIE RNN op input tensor " + fNB + " is not  found in model.");
         }
         fShapeB = model.GetTensorShape(fNB);
         if (fShapeB.size() != 2 && fShapeB.size() != 4) {
            throw std::runtime_error("TMVA SOFIE RNN op input tensor " + fNB + " is not of 2 or 4 dimensions.");
         }
         if (fShapeB.size() == 2) {
            // Broadcasting the bias
            auto original_data = model.GetInitializedTensorData(fNB);
            size_t num_directions = fShapeW[0];
            size_t seq_length = (fAttrLayout == 0)? fShapeX[0] : fShapeX[1];
            size_t batch_size = (fAttrLayout == 0)? fShapeX[1] : fShapeX[0];
            if (fType == "float") {
               float *original_bias = static_cast<float*>(original_data.get());
               float *new_bias = new float[num_directions * seq_length * batch_size * fAttrHiddenSize];
               float sum[fAttrHiddenSize];
               for (size_t direction = 0; direction < num_directions; direction++) {
                  for (size_t h = 0; h < fAttrHiddenSize; h++) {
                     sum[h] = original_bias[direction * 2*fAttrHiddenSize + h]
                        + original_bias[(2 * direction + 1) * fAttrHiddenSize + h];
                  }
                  for (size_t seq = 0; seq < seq_length; seq++) {
                     for (size_t batch = 0; batch < batch_size; batch++) {
                        size_t bias_offset = direction * seq_length * batch_size * fAttrHiddenSize
                           + seq * batch_size * fAttrHiddenSize + batch * fAttrHiddenSize;
                        std::copy(sum, sum + fAttrHiddenSize, new_bias + bias_offset);
                     }
                  }
               }
               std::vector<size_t> new_bias_shape = {num_directions, seq_length, batch_size, fAttrHiddenSize};
               std::shared_ptr<void> new_bias_ptr(new_bias, std::default_delete<float[]>());
               model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), new_bias_shape, new_bias_ptr);
               fShapeB = model.GetTensorShape(fNB);
            }
         }
      }
      if (!fNSequence_lens.empty()) {
         if (!model.CheckIfTensorAlreadyExist(fNSequence_lens)) {
            throw std::runtime_error("TMVA SOFIE RNN Op input tensor " +
                                     fNSequence_lens +
                                     "is not found in model.");
         }
         fShapeSequence_lens = model.GetTensorShape(fNSequence_lens);
         if (fShapeSequence_lens.size() != 1) {
            throw std::runtime_error("TMVA SOFIE RNN Op input tensor " +
                                     fNSequence_lens +
                                     " is not of 1 dimension.");
         }
      }
      if (!fNInitial_h.empty()) {
         if (!model.CheckIfTensorAlreadyExist(fNInitial_h)) {
            throw std::runtime_error("TMVA SOFIE RNN Op input tensor " +
                                     fNInitial_h + " is not found in model.");
         }
         fShapeInitial_h = model.GetTensorShape(fNInitial_h);
         if (fShapeInitial_h.size() != 3) {
            throw std::runtime_error("TMVA SOFIE RNN Op input tensor " +
                                     fNInitial_h + " is not of 3 dimensions.");
         }
      }
      if (!fNY.empty()) {
         fShapeY = ShapeInference({fShapeX, fShapeW})[0];
         if (!model.CheckIfTensorAlreadyExist(fNY)) {
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
         }
      }
      if (!fNY_h.empty()) {
         fShapeY_h = ShapeInference({fShapeX, fShapeW})[1];
         if (!model.CheckIfTensorAlreadyExist(fNY_h)) {
            model.AddIntermediateTensor(fNY_h, model.GetTensorType(fNX), fShapeY_h);
         }
      }
      // Check the attributes
      for (auto &activation : fAttrActivations) {
         if (activation != "Relu" && activation != "Tanh" &&
             activation != "Sigmoid" && activation != "Affine" &&
             activation != "LeakyRelu" && activation != "ThresholdRelu" &&
             activation != "ScaledTanh" && activation != "HardSigmoid" &&
             activation != "Elu" && activation != "Softsign" &&
             activation != "Softplus") {
            throw std::runtime_error("TMVA SOFIE - Activation function " +
                                     activation + " not implemented");
         }
      }
      if (fAttrDirection != "forward" && fAttrDirection != "backward" &&
          fAttrDirection != "bidirectional") {
         throw std::runtime_error(
             "TMVA SOFIE - Invalid RNN direction fAttrDirection = " +
             fAttrDirection);
      }
      if (fAttrHiddenSize != fShapeW[1]) {
         throw std::runtime_error(
             "TMVA SOFIE - fAttrHiddenSize must be equal to " +
             std::to_string(fShapeW[1]));
      }
      if (fAttrLayout > 1) {
         throw std::runtime_error("TMVA SOFIE - Layout fAttrLayout = " +
                                  std::to_string(fAttrLayout) +
                                  " must be 0 (timewise) or 1 (batchwise)");
      }
      if (fAttrActivations.empty()) {
         if (fAttrDirection == "bidirectional") {
            fAttrActivations = {"Tanh", "Tanh"};
         } else {
            fAttrActivations = {"Tanh"};
         }
      }
      // Add needed standard library headers
      model.AddNeededStdLib("cmath");
   }

   std::string Generate(std::string OpName) {
      OpName = "op_" + OpName;
      std::stringstream out;

      size_t seq_length = (fAttrLayout == 0) ? fShapeX[0] : fShapeX[1];
      size_t batch_size = (fAttrLayout == 0) ? fShapeX[1] : fShapeX[0];
      size_t input_size = fShapeX[2];
      size_t num_directions = fShapeW[0];

      // set the input
      if (fAttrLayout == 0) {
         if (fType == "float") {
            out << "\t" << "float *" << OpName << "_input = tensor_" << fNX << ";\n";
         }
      } else {
         if (fType == "float") {
            out << "\t" << "float " << OpName << "_input[" << seq_length * batch_size * input_size << "];\n";
         }
         out << "\t" << "for(size_t seq = 0; seq < " << seq_length << "; seq++) {\n";
         out << "\t" << "\t" << "for(size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
         out << "\t" << "\t" << "\t" << "for(size_t i = 0; i < " << input_size << "; i++) {\n";
         out << "\t" << "\t" << "\t" << "\t" << OpName << "_input[seq * " << batch_size * input_size 
             << " + batch * " << input_size << " + i] = " << "tensor_" << fNX << "[batch * "
             << seq_length * input_size << " + seq * " << input_size << " + i];\n";
         out << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "}\n";
         out << "\t" << "}\n";
      }

      // Set the initial hidden state
      if (!fNInitial_h.empty()) {
         if (fAttrLayout == 0) {
            if (fType == "float") {
               out << "\t" << "float *" << OpName << "_initial_hidden_state = " << " tensor_"
                   << fNInitial_h << ";\n";
            }
         } else {
            if (fType == "float") {
               out << "float " << OpName << "_initial_hidden_state[" << num_directions * batch_size *
                   fAttrHiddenSize << "];\n";
            }
            for (size_t direction = 0; direction < num_directions; direction++) {
               out << "\t" << "for(size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
               out << "\t" << "\t" << "for(size_t h = 0; h < " << fAttrHiddenSize << "; h++) {\n";
               out << "\t" << "\t" << "\t" << OpName << "_initial_hidden_state["
                   << direction * batch_size * fAttrHiddenSize << " + batch * " << fAttrHiddenSize 
                   << " + h] = tensor_" << fNInitial_h << "[batch * " << num_directions * fAttrHiddenSize
                   << " + " << direction * fAttrHiddenSize << " + h];\n";
               out << "\t" << "\t" << "}\n";
               out << "\t" << "}\n";
            }
         }
      }

      if (fType == "float") {
         out << "\t" << "float " << OpName << "_feedforward[" << seq_length * batch_size * fAttrHiddenSize << "];\n";
      }

      // Set the hidden state
      if (fAttrLayout == 0 && !fNY.empty()) {
         if (fType == "float") {
            out << "\t" << "float *" << OpName << "_hidden_state = tensor_" << fNY << ";\n";
         }
      } else {
         if (fType == "float") {
            out << "\t" << "float " << OpName << "_hidden_state[" << seq_length * num_directions *
                batch_size * fAttrHiddenSize << "];\n";
         }
      }

      out << "\t" << "char " << OpName << "_transA = 'N';\n";
      out << "\t" << "char " << OpName << "_transB = 'T';\n";
      out << "\t" << "int " << OpName << "_m = " << seq_length * batch_size << ";\n";
      out << "\t" << "int " << OpName << "_n = " << fAttrHiddenSize << ";\n";
      out << "\t" << "int " << OpName << "_k = " << input_size << ";\n";
      if (fType == "float") {
         out << "\t" << "float " << OpName << "_alpha = 1.;\n";
         out << "\t" << "float " << OpName << "_beta = .0;\n";
      }
      if (!fNB.empty()) {
         out << "\t" << "int " << OpName << "_bias_size = " << seq_length * batch_size * fAttrHiddenSize << ";\n";
         out << "\t" << "int " << OpName << "_incx = 1;\n";
         out << "\t" << "int " << OpName << "_incy = 1;\n";
      }

      for (size_t direction = 0; direction < num_directions; direction++) {
         // feedforward = input * W^T + bias
         if (fType == "float") {
            if (direction == 0) {
               out << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                   << OpName <<"_n, &" << OpName << "_m, &" << OpName << "_k, &" << OpName << "_alpha, tensor_"
                   << fNW << ", &" << OpName << "_k, " << OpName << "_input, &" << OpName << "_k, &"
                  << OpName << "_beta, " << OpName << "_feedforward, &" << OpName << "_n);\n";
            } else {
               out << "\t" << "size_t " << OpName << "_w_offset = " << fAttrHiddenSize * input_size << ";\n";
               out << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                   << OpName <<"_n, &" << OpName << "_m, &" << OpName << "_k, &" << OpName << "_alpha, tensor_"
                  << fNW << " + " << OpName << "_w_offset, &" << OpName << "_k, " << OpName << "_input, &"
                  << OpName << "_k, &" << OpName << "_beta, " << OpName << "_feedforward, &" << OpName << "_n);\n";
            }
         }
         // Add the bias
         if (!fNB.empty()) {
            if (fType == "float") {
               if (direction == 0) {
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                   << fNB << ", &" << OpName << "_incx, " << OpName << "_feedforward, &" << OpName << "_incy);\n";
               } else {
                  out << "\t" << "size_t " << OpName << "_bias_offset = "
                      << seq_length * batch_size * fAttrHiddenSize << ";\n";
                  out << "\t" << "BLAS::saxpy_(&" << OpName << "_bias_size, &" << OpName << "_alpha, tensor_"
                      << fNB << " + " << OpName << "_bias_offset, &" << OpName << "_incx, " << OpName
                      << "_feedforward, &" << OpName << "_incy);\n";
               }
            }
         }

         // Copy feedforward into hidden state
         out << "\t" << "for (size_t seq = 0; seq < " << seq_length << "; seq++) {\n";
         out << "\t" << "\t" << "size_t offset = seq * " << batch_size * fAttrHiddenSize << ";\n";
         out << "\t" << "\t" << "size_t size = " << batch_size * fAttrHiddenSize << ";\n";
         out << "\t" << "\t" << "size_t h_offset = seq * " << num_directions * batch_size * fAttrHiddenSize
             << " + " << direction * batch_size * fAttrHiddenSize << ";\n";
         out << "\t" << "\t" << "std::copy(" << OpName << "_feedforward + offset, " << OpName
             << "_feedforward + offset + size, " << OpName << "_hidden_state + h_offset);\n";
         out << "\t" << "}\n";


         out << "\t" << "for (size_t seq = 0; seq < " << seq_length << "; seq++) {\n";
         if (fAttrDirection == "backward" || direction == 1) {
            out << "\t" << "\t" << "size_t index = " << seq_length - 1 << " - seq;\n";
         } else {
            out << "\t" << "\t" << "size_t index = seq;\n";
         }

         out << "\t" << "\t" << "int m2 = " << batch_size << ";\n";
         out << "\t" << "\t" << "size_t offset = index * " << num_directions * batch_size * fAttrHiddenSize
                << " + " << direction * batch_size * fAttrHiddenSize << ";\n";
         out << "\t" << "\t" << "size_t size = " << batch_size * fAttrHiddenSize << ";\n";
         out << "\t" << "\t" << "if (seq == 0) {\n";
         if (!fNInitial_h.empty()) {
            // hidden_state = hidden_state + initial_hidden_state * R^T
            out << "\t" << "\t" << "\t" << "size_t r_offset = "
                << direction * fAttrHiddenSize * fAttrHiddenSize << ";\n";
            out << "\t" << "\t" << "\t" << "size_t initial_hidden_state_offset = "
                << direction * batch_size * fAttrHiddenSize << ";\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName
                   << "_transA, &" << OpName << "_n, &m2, &" << OpName << "_n, &" << OpName 
                   << "_alpha, tensor_" << fNR << " + r_offset, &" << OpName << "_n, " << OpName
                   << "_initial_hidden_state + initial_hidden_state_offset, &" << OpName << "_n, &"
                   << OpName << "_alpha, " << OpName << "_hidden_state + offset, &" << OpName << "_n);\n";
            }
         }
         out << "\t" << "\t" << "} else {\n";
         // hidden_state = hidden_state + previous_hidden_state * R^T
         out << "\t" << "\t" << "\t" << "size_t r_offset = "
             << direction * fAttrHiddenSize * fAttrHiddenSize << ";\n";
         if (fAttrDirection == "backward" || direction == 1) {
            out << "\t" << "\t" << "\t" << "size_t previous_offset = (index + 1) * "
                << num_directions * batch_size * fAttrHiddenSize
                << " + " << direction * batch_size * fAttrHiddenSize << ";\n";
         } else {
            out << "\t" << "\t" << "\t" << "size_t previous_offset = (seq - 1) * "
                << num_directions * batch_size * fAttrHiddenSize
                << " + " << direction * batch_size * fAttrHiddenSize << ";\n";
         }
         if (fType == "float") {
            out << "\t" << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &"
                << OpName << "_n, &m2, &" << OpName << "_n, &" << OpName << "_alpha, tensor_" << fNR
                << " + r_offset, &" << OpName << "_n, " << OpName << "_hidden_state + previous_offset, &"
                << OpName << "_n, &" << OpName << "_alpha, " << OpName << "_hidden_state + offset, &"
                << OpName << "_n);\n";
         }
         out << "\t" << "\t" << "}\n";

         // Clip the elements of the hidden state into the range [-fAttrClip, fAttrClip]
         if (fAttrClip > .0) {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float x = (" << OpName << "_hidden_state[i] > " << -fAttrClip
                   << ") ? " << OpName << "_hidden_state[i] : " << -fAttrClip << ";\n";
            }
            out << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = (x < " << fAttrClip
                << ") ? x : " << fAttrClip << ";\n";
            out << "\t" << "\t" << "}\n";
         }

         // Apply the activation function to the hidden state
         if (fAttrActivations[direction] == "Relu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_hidden_state[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = 0.;\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction] == "Tanh") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float ex = std::exp(-2 * " << OpName << "_hidden_state[i]);\n";
            }
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = (1. - ex) / (1. + ex);\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction] == "Sigmoid") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = 1. / (1. + std::exp(-" << OpName
                << "_hidden_state[i]));\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction] == "Affine") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = " << fAttrActivationAlpha[direction]
                << " * " << OpName << "_hidden_state[i] + " << fAttrActivationBeta[direction] << ";\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction] == "ScaledTanh") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float ex = std::exp(-2 * " << fAttrActivationBeta[direction]
                   << " * "<< OpName << "_hidden_state[i]);\n";
               }
               out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = " << fAttrActivationAlpha[direction]
                   << " * (1. - ex) / (1. + ex);\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction] == "HardSigmoid") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            if (fType == "float") {
               out << "\t" << "\t" << "\t" << "float a = " << fAttrActivationAlpha[direction] << " * "
                   << OpName << "_hidden_state[i] + " << fAttrActivationBeta[direction] << ";\n";
               out << "\t" << "\t" << "\t" << "float b = (a > 0.) ? a : 0.;\n";
            }
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = (b < 1.) ? b : 1.;\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction] == "LeakyRelu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_hidden_state[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = " << fAttrActivationAlpha[direction]
                << " * " << OpName << "_hidden_state[i];\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction] == "ThresholdRelu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_hidden_state[i] < "
                << fAttrActivationAlpha[direction] << ")\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = 0.;\n";
            out << "\t" << "\t" << "}";
         } else if (fAttrActivations[direction] == "Elu") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            out << "\t" << "\t" << "\t" << "if (" << OpName << "_hidden_state[i] < 0.)\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = " << fAttrActivationAlpha[direction]
                << " * std::exp(" << OpName << "_hidden_state[i] - 1.);\n";
            out << "\t" << "\t" << "}\n";
         } else if (fAttrActivations[direction] == "Softsign") {
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = " << OpName 
                << "_hidden_state[i] / (1. + abs(" << OpName << "_hidden_state[i]));\n";
            out << "\t" << "\t" << "}\n";
         } else { // fAttrActivations[direction] = Softplus
            out << "\t" << "\t" << "for (size_t i = offset; i < offset + size; i++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[i] = log(1. + std::exp("
                << OpName << "_hidden_state[i]));\n";
            out << "\t" << "\t" << "}\n";
            out << "\t" << "}\n";
         }
         out << "\t" << "}\n";
      }

      // Padding the hidden state for RNN with different sequence lengths
      if (!fNSequence_lens.empty()) {
         out << "\t" << "for (size_t seq = 0; seq < " << seq_length << "; seq++) {\n";
         out << "\t" << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
         out << "\t" << "\t" << "\t" << "if (seq >= tensor_" << fNSequence_lens << "[batch]) {\n";
         for (size_t direction = 0; direction < num_directions; direction++) {
            out << "\t" << "\t" << "\t" << "\t" << "\t" << "for (size_t h = 0; h < " << fAttrHiddenSize << "; h++) {\n";
            out << "\t" << "\t" << "\t" << "\t" << "\t" << "\t" << OpName << "_hidden_state[seq * "
                << num_directions * batch_size * fAttrHiddenSize + direction * batch_size * fAttrHiddenSize
                << " + batch * " << fAttrHiddenSize << " + h] = 0.;\n";
            out << "\t" << "\t" << "\t" << "\t" << "\t" << "}\n";
         }
         out << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "}\n";
         out << "\t" << "}\n";
      }

      // Copy the hidden state into y and y_h
      if (fAttrLayout == 0) {
         if (!fNY_h.empty()) {
            if (fNSequence_lens.empty()) {
               size_t yh_size = batch_size * fAttrHiddenSize;
               if (fAttrDirection == "backward") {
                  out << "\t" << "std::copy(" << OpName << "_hidden_state, " << OpName << "_hidden_state + "
                      << yh_size << ", tensor_" << fNY_h << ");\n";
               } else {
                  size_t offset = (seq_length - 1) * num_directions * batch_size * fAttrHiddenSize;
                  out << "\t" << "std::copy(" << OpName << "_hidden_state + " << offset << ", " << OpName
                      << "_hidden_state + " << offset << " + " << yh_size << ", tensor_" << fNY_h << ");\n";
               }
               if (num_directions == 2) {
                  out << "\t" << "std::copy(" << OpName << "_hidden_state + " << yh_size << ", " << OpName
                      << "_hidden_state + " << 2 * yh_size << ", tensor_" << fNY_h << " + " << yh_size << ");\n";
               }
            } else { // RNN with different sequence lengths
               if (fAttrDirection == "backward") {
                  out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
                  out << "\t" << "\t" << "size_t offset = batch * " << fAttrHiddenSize << ";\n";
                  out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                      << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + offset);\n";
                  out << "\t" << "}\n";
               } else {
                  out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
                  out << "\t" << "\t" << "size_t seq = " << "tensor_" << fNSequence_lens << "[batch] - 1;\n";
                  out << "\t" << "\t" << "size_t offset = seq * " << num_directions * batch_size * fAttrHiddenSize
                      << " + batch * " << fAttrHiddenSize << ";\n";
                  out << "\t" << "\t" << "size_t yh_offset = batch * " << fAttrHiddenSize << ";\n";
                  out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                      << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + yh_offset);\n";
                  out << "\t" << "}\n";
               }
               if (num_directions == 2) {
                  out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
                  out << "\t" << "\t" << "size_t offset = " << batch_size * fAttrHiddenSize
                      << " + batch * " << fAttrHiddenSize << ";\n";
                  out << "\t" << "\t" << "size_t yh_offset = " << batch_size * fAttrHiddenSize
                      << " + batch * " << fAttrHiddenSize << ";\n";
                  out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                      << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + yh_offset);\n";
                  out << "\t" << "}\n";
               }
            }
         }
      } else { // fAttrLayout=1
         if (!fNY.empty()) {
            for (size_t direction = 0; direction < num_directions; direction++) {
               out << "\t" << "for (size_t seq = 0; seq < " << seq_length << "; seq++) {\n";
               out << "\t" << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
               out << "\t" << "\t" << "\t" << "size_t offset = seq * " << num_directions * batch_size * fAttrHiddenSize
                   << " + " << direction * batch_size * fAttrHiddenSize << " + batch * " << fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "\t" << "size_t y_offset = batch * " << seq_length * num_directions * fAttrHiddenSize
                   << " + seq * " << num_directions * fAttrHiddenSize << " + " << direction * fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                   << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY << " + y_offset);\n";
               out << "\t" << "\t" << "}\n";
               out << "\t" << "}\n";
            }
         }
         if (!fNY_h.empty()) {
            if (fAttrDirection == "backward") {
               out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
               out << "\t" << "\t" << "size_t offset = batch * " << fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "size_t yh_offset = batch * " << num_directions * fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                   << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + yh_offset);\n";
               out << "\t" << "}\n";
            } else {
               out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
               if (fNSequence_lens.empty()) {
                  out << "\t" << "\t" << "size_t seq = " << seq_length - 1 << ";\n";
               } else {
                  out << "\t" << "\t" << "size_t seq = " << "tensor_" << fNSequence_lens << "[batch] - 1;\n";
               }
               out << "\t" << "\t" << "size_t offset = seq * " << num_directions * batch_size * fAttrHiddenSize
                   << " + batch * " << fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "size_t yh_offset = batch * " << num_directions * fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                   << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + yh_offset);\n";
               out << "\t" << "}\n";
            }
            if (num_directions == 2) {
               out << "\t" << "for (size_t batch = 0; batch < " << batch_size << "; batch++) {\n";
               out << "\t" << "\t" << "size_t offset = " << batch_size * fAttrHiddenSize << " + batch * "
                   << fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "size_t yh_offset = batch * " << num_directions * fAttrHiddenSize << " + "
                   << fAttrHiddenSize << ";\n";
               out << "\t" << "\t" << "std::copy(" << OpName << "_hidden_state + offset, " << OpName
                   << "_hidden_state + offset + " << fAttrHiddenSize << ", tensor_" << fNY_h << " + yh_offset);\n";
               out << "\t" << "}\n";
            }
         }
      }

      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
