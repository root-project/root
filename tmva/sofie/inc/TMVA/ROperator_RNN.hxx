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

/*! \brief Recurrent Neural Network operator
 *
 * Inference code generation for one-layer vanilla RNN. Supports forward, reverse and bidirectional RNNs.
 * See the <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#RNN">ONNX documentation</a>
 * for details about the supported RNN architectures.
 */
template <typename T> class ROperator_RNN final : public ROperator {
 private:
   std::vector<float> fAttrActivationAlpha;   ///< Scaling values used by some activation functions
   std::vector<float> fAttrActivationBeta;    ///< Scaling values used by some activation functions
   std::vector<std::string> fAttrActivations; ///< Activation functions
   float fAttrClip;                           ///< Clip threshold
   std::string fAttrDirection;                ///< Direction of processing
   size_t fAttrHiddenSize;                    ///< Number of the hidden layers
   size_t fAttrLayout;                        ///< Data layout

   std::string fNX;                           ///< Name of the input
   std::string fNW;                           ///< Name of the weights
   std::string fNR;                           ///< Name of the recurrence
   std::string fNB;                           ///< Name of the bias
   std::string fNSequence_lens;               ///< Name of the length of the sequences
   std::string fNInitial_h;                   ///< Name of the initial value of the hidden states
   std::string fNY;                           ///< Name of the output
   std::string fNY_h;                         ///< Name of the last sequence of the output

   std::vector<size_t> fShapeX;               ///< Shape of the input
   std::vector<size_t> fShapeW;               ///< Shape of the weights
   std::vector<size_t> fShapeR;               ///< Shape of the recurrence
   std::vector<size_t> fShapeB;               ///< Shape of the bias
   std::vector<size_t> fShapeSequence_lens;   ///< Shape of the length of the sequences
   std::vector<size_t> fShapeInitial_h;       ///< Shape of the initial value of the hidden states
   std::vector<size_t> fShapeY;               ///< Shape of the output
   std::vector<size_t> fShapeY_h;             ///< Shape of the last sequence of the output

   std::string fType; ///< Type of the tensors

 public:
   /*! Default constructor of ROperator_RNN */
   ROperator_RNN() {}

   /*! \brief Constructor of ROperator_RNN from the attributes
    *
    * \param activation_alpha scaling values used by some activation functions
    * \param activation_beta scaling values used by some activation functions
    * \param activations activation functions
    * \param clip clip threshold
    * \param direction direction of processing of the sequneces
    * \param hidden_size number of hidden layers
    * \param layout data layout
    * \param nameX name of the input tensor
    * \param nameW name of the weight tensor
    * \param nameR name of the recurrence tensor
    * \param nameB name of the bias tensor
    * \param nameSequence_lens name of the length of the sequences
    * \param nameInitial_h name of the initial value of the hidden states
    * \param nameY name of the output
    * \param nameY_h name of the last sequence of the output
    */
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

   /*! \brief Infers the type of the output tensors
    *
    * \param input type of the input tensors
    */
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input);

   /*! \brief Infers the shape of the output tensors
    *
    * \param input shape of the input tensors
    */
   std::vector<std::vector<size_t>>
   ShapeInference(std::vector<std::vector<size_t>> input);

   /*! \brief Initialize the model
    *
    * \param model Model
    */
   void Initialize(RModel &model);

   /*! \brief Generates the inference code
    *
    * \param OpName name of the operator
    */
   std::string Generate(std::string OpName);
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

// Implementation of the ROperator_RNN class
#include "TMVA/ROperator_RNN.icc"

#endif
