#ifndef TMVA_SOFIE_ROPERATOR_CONVTRANSPOSE_HXX
#define TMVA_SOFIE_ROPERATOR_CONVTRANSPOSE_HXX

#include <TMVA/SOFIE_common.hxx>
#include <TMVA/ROperator.hxx>
#include <TMVA/RModel.hxx>

#include <memory>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

/*! \brief Transposed Convolution operator
 *
 * Inference code generation for a transposed convolution layer.
 * See the <a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#convtranspose">ONNX documentation</a> for
 * details about the transposed conv layer.
 */
template <typename T>
class ROperator_ConvTranspose final : public ROperator {
private:
   std::string fAttrAutopad;
   std::vector<size_t> fAttrDilations;
   size_t fAttrGroup;
   std::vector<size_t> fAttrKernelShape;
   std::vector<size_t> fAttrOutputPadding;
   std::vector<size_t> fAttrOutputShape;
   std::vector<size_t> fAttrPads;
   std::vector<size_t> fAttrStrides;

   std::string fNX;
   std::string fNW;
   std::string fNB;
   std::string fNBroadcastedB;
   std::string fNY;

   std::string fConvK;
   std::string fImcol;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeW;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeY;

   std::string fType;

   size_t fDim; // dimension of the convolution

public:
   /*! Default constructor of ROperator_ConvTranspose */
   ROperator_ConvTranspose() {}

   /*! \brief Constructor of ROperator_ConvTranspose from the attributes
    *
    * \param autopad padding
    * \param dilations dilations of the kernel
    * \param group number of groups
    * \param kernelShape shape of the kernel
    * \param outputPadding padding of the output
    * \param outputShape shape of the output
    * \param pads padding of the input
    * \param strides strides
    * \param nameX name of the input
    * \param nameW name of the weight
    * \param nameB name of the bias
    * \param nameY name of the output
    */
   ROperator_ConvTranspose(std::string autopad, std::vector<size_t> dilations, size_t group,
                           std::vector<size_t> kernelShape, std::vector<size_t> outputPadding,
                           std::vector<size_t> outputShape, std::vector<size_t> pads, std::vector<size_t> strides,
                           std::string nameX, std::string nameW, std::string nameB, std::string nameY)
      : fAttrAutopad(autopad), fAttrDilations(dilations), fAttrGroup(group), fAttrKernelShape(kernelShape),
        fAttrOutputPadding(outputPadding), fAttrOutputShape(outputShape), fAttrPads(pads), fAttrStrides(strides),
        fNX(UTILITY::Clean_name(nameX)), fNW(UTILITY::Clean_name(nameW)), fNB(UTILITY::Clean_name(nameB)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = { fNX, fNW };
      fOutputTensorNames = { fNY };
      if (!fNB.empty()) {
         fInputTensorNames.emplace_back(fNB);
      }

      if (std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a Conv operator");
      }
   }

   /*! \brief Infers the type of the output tensor
    * \param input type of the input tensors
    */
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override
   {
      ETensorType out = input[0];
      return {out};
   }

   /*! \brief Infers the shape of the input tensors
    * \param input shape of the input tensors
    */
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> /*input*/) override;

   /*! \brief Initialize the model
    * \param model Model
    */
   void Initialize(RModel &) override;

   /*! \brief Generate code for initializing the op
    */
   std::string GenerateInitCode() override;

   /*! \brief Generate the inference code
    * \param opName name of the operator
    */
   std::string Generate(std::string opName) override;

   /*! \brief Returns the blas routines needed to compile the generated code
    */
   std::vector<std::string> GetBlasRoutines() override { return { std::string("Gemm"), std::string("Axpy") }; }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

// Implementation of the ROperator_ConvTranspose class
#include "TMVA/ROperator_ConvTranspose.icc"

#endif
