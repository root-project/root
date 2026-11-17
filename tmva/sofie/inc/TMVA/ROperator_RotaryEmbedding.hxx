#ifndef TMVA_SOFIE_ROPERATOR_ROTARYEMBEDDING
#define TMVA_SOFIE_ROPERATOR_ROTARYEMBEDDING

#include "TMVA/RModel.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/SOFIE_common.hxx"

#include <sstream>
#include <string>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

/// ONNX RotaryEmbedding (opset 23), the rotary positional embedding (RoPE) of https://arxiv.org/abs/2104.09864.
///
/// X is either 4d [batch, num_heads, seq, head_size] or 3d [batch, seq, num_heads * head_size] (then the
/// num_heads attribute is required). The first rotary_embedding_dim (R, default head_size) elements of every
/// head are rotated pairwise, the remaining ones are copied:
///    y1 = cos * x1 - sin * x2,   y2 = sin * x1 + cos * x2
/// where (x1, x2) are the two halves of the rotated part, or its even and odd elements if interleaved = 1.
/// cos_cache and sin_cache are [batch, seq, R/2], or [max_position, R/2] indexed with position_ids [batch, seq].
template <typename T>
class ROperator_RotaryEmbedding final : public ROperator {
private:
   bool fInterleaved;
   size_t fAttrRotaryDim;
   size_t fAttrNumHeads;

   std::string fNX;
   std::string fNCos;
   std::string fNSin;
   std::string fNPositionIds; // optional
   std::string fNY;

   std::vector<Dim> fShapeX;
   bool fIs3d = false;
   Dim fBatch;
   Dim fSeq;
   size_t fNumHeads = 0;
   size_t fHeadSize = 0;
   size_t fRotaryDim = 0;

public:
   ROperator_RotaryEmbedding() {}

   ROperator_RotaryEmbedding(bool interleaved, size_t rotaryDim, size_t numHeads, const std::string &nameX,
                             const std::string &nameCos, const std::string &nameSin, const std::string &namePositionIds,
                             const std::string &nameY)
      : fInterleaved(interleaved),
        fAttrRotaryDim(rotaryDim),
        fAttrNumHeads(numHeads),
        fNX(UTILITY::Clean_name(nameX)),
        fNCos(UTILITY::Clean_name(nameCos)),
        fNSin(UTILITY::Clean_name(nameSin)),
        fNPositionIds(UTILITY::Clean_name(namePositionIds)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = {fNX, fNCos, fNSin};
      if (!fNPositionIds.empty())
         fInputTensorNames.emplace_back(fNPositionIds);
      fOutputTensorNames = {fNY};
   }

   void Initialize(RModel &model) override
   {
      for (const std::string &name : {fNX, fNCos, fNSin, fNPositionIds}) {
         if (!name.empty() && !model.CheckIfTensorAlreadyExist(name))
            throw std::runtime_error("TMVA::SOFIE - RotaryEmbedding - Tensor " + name + " not found.");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      // the last dimension (head_size or hidden_size) must be known to split the heads and the rotated part
      if (fShapeX.size() == 4) {
         fIs3d = false;
         fBatch = fShapeX[0];
         fSeq = fShapeX[2];
         if (fShapeX[1].isParam || fShapeX[3].isParam)
            throw std::runtime_error("TMVA::SOFIE - RotaryEmbedding - the number of heads and the head size of " + fNX +
                                     " must be known, got shape " + ConvertDimShapeToString(fShapeX));
         fNumHeads = fShapeX[1].dim;
         fHeadSize = fShapeX[3].dim;
         if (fAttrNumHeads != 0 && fAttrNumHeads != fNumHeads)
            throw std::runtime_error("TMVA::SOFIE - RotaryEmbedding - num_heads attribute " +
                                     std::to_string(fAttrNumHeads) + " does not match the input shape " +
                                     ConvertDimShapeToString(fShapeX));
      } else if (fShapeX.size() == 3) {
         fIs3d = true;
         fBatch = fShapeX[0];
         fSeq = fShapeX[1];
         if (fAttrNumHeads == 0)
            throw std::runtime_error("TMVA::SOFIE - RotaryEmbedding - num_heads attribute is required for a 3d input");
         if (fShapeX[2].isParam || fShapeX[2].dim % fAttrNumHeads != 0)
            throw std::runtime_error("TMVA::SOFIE - RotaryEmbedding - hidden size of " + fNX + " with shape " +
                                     ConvertDimShapeToString(fShapeX) + " must be a known multiple of num_heads " +
                                     std::to_string(fAttrNumHeads));
         fNumHeads = fAttrNumHeads;
         fHeadSize = fShapeX[2].dim / fAttrNumHeads;
      } else {
         throw std::runtime_error("TMVA::SOFIE - RotaryEmbedding - input " + fNX + " must be 3d or 4d, got shape " +
                                  ConvertDimShapeToString(fShapeX));
      }

      fRotaryDim = (fAttrRotaryDim == 0) ? fHeadSize : fAttrRotaryDim;
      if (fRotaryDim > fHeadSize || fRotaryDim % 2 != 0)
         throw std::runtime_error("TMVA::SOFIE - RotaryEmbedding - rotary_embedding_dim " + std::to_string(fRotaryDim) +
                                  " must be even and not larger than the head size " + std::to_string(fHeadSize));

      // the caches are [batch, seq, R/2], or [max_position, R/2] when indexed with position_ids
      for (auto &name : {fNCos, fNSin}) {
         auto shape = model.GetDimTensorShape(name);
         const size_t expectedRank = fNPositionIds.empty() ? 3 : 2;
         if (shape.size() != expectedRank || shape.back().isParam || shape.back().dim != fRotaryDim / 2)
            throw std::runtime_error(
               "TMVA::SOFIE - RotaryEmbedding - cache " + name + " has shape " + ConvertDimShapeToString(shape) +
               ", expected rank " + std::to_string(expectedRank) +
               " with last dimension rotary_embedding_dim / 2 = " + std::to_string(fRotaryDim / 2));
         // without position_ids the cache is read per (batch, token): it must not be smaller than X
         if (fNPositionIds.empty()) {
            const bool batchOk = shape[0].isParam || fBatch.isParam || shape[0].dim == fBatch.dim;
            const bool seqOk = shape[1].isParam || fSeq.isParam || shape[1].dim == fSeq.dim;
            if (!batchOk || !seqOk)
               throw std::runtime_error("TMVA::SOFIE - RotaryEmbedding - cache " + name + " has shape " +
                                        ConvertDimShapeToString(shape) +
                                        " which does not match the batch size and sequence length of input " +
                                        ConvertDimShapeToString(fShapeX));
         }
      }
      if (!fNPositionIds.empty()) {
         if (model.GetTensorType(fNPositionIds) != ETensorType::INT64)
            throw std::runtime_error("TMVA::SOFIE - RotaryEmbedding - position_ids " + fNPositionIds +
                                     " must be of type int64");
         if (model.GetDimTensorShape(fNPositionIds).size() != 2)
            throw std::runtime_error("TMVA::SOFIE - RotaryEmbedding - position_ids " + fNPositionIds +
                                     " must have shape [batch, sequence_length]");
      }

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeX);
      if (model.Verbose()) {
         std::cout << "RotaryEmbedding : " << fNX << " -> " << fNY << " shape " << ConvertDimShapeToString(fShapeX)
                   << " heads " << fNumHeads << " head size " << fHeadSize << " rotary dim " << fRotaryDim
                   << (fInterleaved ? " interleaved" : "") << std::endl;
      }
   }

   std::string Generate(std::string opName) override
   {
      opName = "op_" + opName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA::SOFIE RotaryEmbedding operator " + opName +
                                  " called to generate without being initialized first.");
      }
      const std::string batch = fBatch.GetVal();
      const std::string seq = fSeq.GetVal();
      const size_t halfDim = fRotaryDim / 2;

      std::stringstream out;
      out << "//---- RotaryEmbedding operator " << opName << "\n";
      out << SP << "for (size_t b = 0; b < " << batch << "; b++) {\n";
      out << SP << SP << "for (size_t s = 0; s < " << seq << "; s++) {\n";
      // row of the cos/sin caches for this token
      if (fNPositionIds.empty())
         out << SP << SP << SP << "const size_t cacheOffset = (b * " << seq << " + s) * " << halfDim << ";\n";
      else
         out << SP << SP << SP << "const size_t cacheOffset = tensor_" << fNPositionIds << "[b * " << seq << " + s] * "
             << halfDim << ";\n";
      out << SP << SP << SP << "for (size_t h = 0; h < " << fNumHeads << "; h++) {\n";
      // offset of the head vector of this token in X and Y
      if (fIs3d)
         out << SP << SP << SP << SP << "const size_t offset = ((b * " << seq << " + s) * " << fNumHeads << " + h) * "
             << fHeadSize << ";\n";
      else
         out << SP << SP << SP << SP << "const size_t offset = ((b * " << fNumHeads << " + h) * " << seq << " + s) * "
             << fHeadSize << ";\n";
      out << SP << SP << SP << SP << "for (size_t i = 0; i < " << halfDim << "; i++) {\n";
      const std::string i1 = fInterleaved ? "offset + 2 * i" : "offset + i";
      const std::string i2 = fInterleaved ? "offset + 2 * i + 1" : "offset + i + " + std::to_string(halfDim);
      out << SP << SP << SP << SP << SP << "const " << TensorType<T>::Name() << " c = tensor_" << fNCos
          << "[cacheOffset + i];\n";
      out << SP << SP << SP << SP << SP << "const " << TensorType<T>::Name() << " sn = tensor_" << fNSin
          << "[cacheOffset + i];\n";
      out << SP << SP << SP << SP << SP << "const " << TensorType<T>::Name() << " x1 = tensor_" << fNX << "[" << i1
          << "];\n";
      out << SP << SP << SP << SP << SP << "const " << TensorType<T>::Name() << " x2 = tensor_" << fNX << "[" << i2
          << "];\n";
      out << SP << SP << SP << SP << SP << "tensor_" << fNY << "[" << i1 << "] = c * x1 - sn * x2;\n";
      out << SP << SP << SP << SP << SP << "tensor_" << fNY << "[" << i2 << "] = sn * x1 + c * x2;\n";
      out << SP << SP << SP << SP << "}\n";
      // the part of the head beyond rotary_embedding_dim is not rotated
      if (fRotaryDim < fHeadSize) {
         out << SP << SP << SP << SP << "for (size_t i = " << fRotaryDim << "; i < " << fHeadSize << "; i++)\n";
         out << SP << SP << SP << SP << SP << "tensor_" << fNY << "[offset + i] = tensor_" << fNX << "[offset + i];\n";
      }
      out << SP << SP << SP << "}\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
