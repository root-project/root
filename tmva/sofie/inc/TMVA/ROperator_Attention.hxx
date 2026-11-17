#ifndef TMVA_SOFIE_ROPERATOR_ATTENTION
#define TMVA_SOFIE_ROPERATOR_ATTENTION

#include "TMVA/RModel.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/SOFIE_common.hxx"

#include <sstream>
#include <string>
#include <tuple>
#include <utility>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

/// ONNX Attention (opset 23): scaled dot-product attention
///    Y = softmax(softcap(Q K^T * scale + bias)) V
/// for multi-head, grouped-query and multi-query attention.
///
/// Q, K and V are either 4d, [batch, heads, seq, head_size], or 3d, [batch, seq, heads * head_size] with the
/// q_num_heads and kv_num_heads attributes. Query head h attends to key/value head h / (q_heads / kv_heads).
/// The bias is built from the optional attn_mask, broadcastable to [batch, q_heads, q_seq, kv_seq] (a bool mask
/// keeps the positions that are true, a float mask is added; key positions beyond its last dimension are masked
/// out), and from is_causal, which masks the keys j > i of query i. The scale defaults to 1/sqrt(head_size).
///
/// Not supported yet: the past_key/past_value inputs with the present_key/present_value outputs, the
/// nonpad_kv_seqlen input and the qk_matmul_output output.
template <typename T>
class ROperator_Attention final : public ROperator {
private:
   bool fIsCausal;
   float fAttrScale; // 0 means the default 1/sqrt(head_size)
   float fSoftcap;
   size_t fAttrQNumHeads;
   size_t fAttrKVNumHeads;

   std::string fNQ;
   std::string fNK;
   std::string fNV;
   std::string fNMask; // optional
   std::string fNY;

   bool fIs3d = false;
   std::vector<Dim> fShapeQ;
   std::vector<Dim> fShapeY;
   Dim fBatch;
   Dim fQSeq;
   Dim fKVSeq;
   size_t fQHeads = 0;
   size_t fKVHeads = 0;
   size_t fHeadSize = 0;  // of Q and K
   size_t fVHeadSize = 0; // of V and Y
   float fScale = 0;

   bool fMaskIsBool = false;
   std::vector<Dim> fShapeMask; // left-padded with 1 to rank 4

   // number of heads and head size of a 3d [batch, seq, heads * head_size] or 4d [batch, heads, seq, head_size]
   // tensor; the sequence length is returned in seq
   std::pair<size_t, size_t> GetHeads(RModel &model, const std::string &name, size_t attrHeads, Dim &seq) const
   {
      auto shape = model.GetDimTensorShape(name);
      if (shape.size() != (fIs3d ? 3u : 4u))
         throw std::runtime_error("TMVA::SOFIE - Attention - Q, K and V must all be 3d or all be 4d, got " + name +
                                  " with shape " + ConvertDimShapeToString(shape));
      if (fIs3d) {
         seq = shape[1];
         if (attrHeads == 0)
            throw std::runtime_error(
               "TMVA::SOFIE - Attention - q_num_heads and kv_num_heads attributes are required for 3d inputs");
         if (shape[2].isParam || shape[2].dim % attrHeads != 0)
            throw std::runtime_error("TMVA::SOFIE - Attention - hidden size of " + name + " with shape " +
                                     ConvertDimShapeToString(shape) + " must be a known multiple of the number " +
                                     "of heads " + std::to_string(attrHeads));
         return {attrHeads, shape[2].dim / attrHeads};
      }
      seq = shape[2];
      if (shape[1].isParam || shape[3].isParam)
         throw std::runtime_error("TMVA::SOFIE - Attention - the number of heads and the head size of " + name +
                                  " must be known, got shape " + ConvertDimShapeToString(shape));
      if (attrHeads != 0 && attrHeads != shape[1].dim)
         throw std::runtime_error("TMVA::SOFIE - Attention - num_heads attribute " + std::to_string(attrHeads) +
                                  " does not match the shape of " + name + " " + ConvertDimShapeToString(shape));
      return {shape[1].dim, shape[3].dim};
   }

public:
   ROperator_Attention() {}

   ROperator_Attention(bool isCausal, float scale, float softcap, size_t qNumHeads, size_t kvNumHeads,
                       const std::string &nameQ, const std::string &nameK, const std::string &nameV,
                       const std::string &nameMask, const std::string &nameY)
      : fIsCausal(isCausal),
        fAttrScale(scale),
        fSoftcap(softcap),
        fAttrQNumHeads(qNumHeads),
        fAttrKVNumHeads(kvNumHeads),
        fNQ(UTILITY::Clean_name(nameQ)),
        fNK(UTILITY::Clean_name(nameK)),
        fNV(UTILITY::Clean_name(nameV)),
        fNMask(UTILITY::Clean_name(nameMask)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = {fNQ, fNK, fNV};
      if (!fNMask.empty())
         fInputTensorNames.emplace_back(fNMask);
      fOutputTensorNames = {fNY};
   }

   void Initialize(RModel &model) override
   {
      for (const std::string &name : {fNQ, fNK, fNV, fNMask}) {
         if (!name.empty() && !model.CheckIfTensorAlreadyExist(name))
            throw std::runtime_error("TMVA::SOFIE - Attention - Tensor " + name + " not found.");
      }
      fShapeQ = model.GetDimTensorShape(fNQ);
      if (fShapeQ.size() != 3 && fShapeQ.size() != 4)
         throw std::runtime_error("TMVA::SOFIE - Attention - Q must be 3d or 4d, got shape " +
                                  ConvertDimShapeToString(fShapeQ));
      fIs3d = fShapeQ.size() == 3;
      fBatch = fShapeQ[0];

      Dim kSeq, vSeq;
      std::tie(fQHeads, fHeadSize) = GetHeads(model, fNQ, fAttrQNumHeads, fQSeq);
      auto [kHeads, kHeadSize] = GetHeads(model, fNK, fAttrKVNumHeads, kSeq);
      auto [vHeads, vHeadSize] = GetHeads(model, fNV, fAttrKVNumHeads, vSeq);
      fKVHeads = kHeads;
      fVHeadSize = vHeadSize;
      fKVSeq = kSeq;
      if (kHeadSize != fHeadSize)
         throw std::runtime_error("TMVA::SOFIE - Attention - Q and K must have the same head size, got " +
                                  std::to_string(fHeadSize) + " and " + std::to_string(kHeadSize));
      if (vHeads != fKVHeads || kSeq != vSeq)
         throw std::runtime_error("TMVA::SOFIE - Attention - K and V must have the same number of heads and "
                                  "sequence length");
      if (fKVHeads == 0 || fQHeads % fKVHeads != 0)
         throw std::runtime_error("TMVA::SOFIE - Attention - the number of query heads " + std::to_string(fQHeads) +
                                  " must be a multiple of the number of key/value heads " + std::to_string(fKVHeads));
      for (const std::string &name : {fNK, fNV}) {
         auto shape = model.GetDimTensorShape(name);
         if (!shape[0].isParam && !fBatch.isParam && shape[0].dim != fBatch.dim)
            throw std::runtime_error("TMVA::SOFIE - Attention - " + name + " has shape " +
                                     ConvertDimShapeToString(shape) + " with a batch size different from Q " +
                                     ConvertDimShapeToString(fShapeQ));
      }
      fScale = (fAttrScale != 0) ? fAttrScale : 1.f / std::sqrt(static_cast<float>(fHeadSize));

      if (!fNMask.empty()) {
         const ETensorType maskType = model.GetTensorType(fNMask);
         fMaskIsBool = maskType == ETensorType::BOOL;
         if (!fMaskIsBool && maskType != model.GetTensorType(fNQ))
            throw std::runtime_error("TMVA::SOFIE - Attention - attn_mask " + fNMask + " must be bool or of the " +
                                     "type of Q, got " + ConvertTypeToString(maskType));
         fShapeMask = model.GetDimTensorShape(fNMask);
         if (fShapeMask.empty() || fShapeMask.size() > 4)
            throw std::runtime_error("TMVA::SOFIE - Attention - attn_mask must have rank 1 to 4, got shape " +
                                     ConvertDimShapeToString(fShapeMask));
         fShapeMask.insert(fShapeMask.begin(), 4 - fShapeMask.size(), Dim{1});
         // the leading axes broadcast to [batch, q_heads, q_seq]; the last one may be shorter than kv_seq
         const std::vector<Dim> target = {fBatch, Dim{fQHeads}, fQSeq};
         for (size_t i = 0; i < 3; i++) {
            const Dim &m = fShapeMask[i];
            if (!m.isParam && m.dim == 1)
               continue; // broadcast axis
            const bool matches = (m.isParam || target[i].isParam) ? !(m != target[i]) : m.dim == target[i].dim;
            if (!matches)
               throw std::runtime_error("TMVA::SOFIE - Attention - attn_mask with shape " +
                                        ConvertDimShapeToString(model.GetDimTensorShape(fNMask)) +
                                        " cannot be broadcast to [batch, q_num_heads, q_sequence_length, *]");
         }
         if (fShapeMask[3].isParam && fShapeMask[3] != fKVSeq)
            throw std::runtime_error("TMVA::SOFIE - Attention - the last dimension of attn_mask must be known or "
                                     "equal to the key/value sequence length");
      }

      if (fIs3d)
         fShapeY = {fBatch, fQSeq, Dim{fQHeads * fVHeadSize}};
      else
         fShapeY = {fBatch, Dim{fQHeads}, fQSeq, Dim{fVHeadSize}};
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNQ), fShapeY);
      model.AddNeededStdLib("algorithm");
      model.AddNeededStdLib("cmath");
      model.AddNeededStdLib("limits");
      if (model.Verbose()) {
         std::cout << "Attention : " << fNQ << ", " << fNK << ", " << fNV << " -> " << fNY << " shape "
                   << ConvertDimShapeToString(fShapeY) << " q heads " << fQHeads << " kv heads " << fKVHeads
                   << " scale " << fScale << (fIsCausal ? " causal" : "") << std::endl;
      }
   }

   std::string Generate(std::string opName) override
   {
      opName = "op_" + opName;
      if (fShapeQ.empty()) {
         throw std::runtime_error("TMVA::SOFIE Attention operator " + opName +
                                  " called to generate without being initialized first.");
      }
      const std::string batch = fBatch.GetVal();
      const std::string qSeq = fQSeq.GetVal();
      const std::string kvSeq = fKVSeq.GetVal();
      const size_t groupSize = fQHeads / fKVHeads;

      // offset of the head vector of token t of head h in a 3d or 4d tensor with the given layout
      auto offset = [&](const std::string &h, const std::string &t, const std::string &seq, size_t heads,
                        size_t headSize) {
         if (fIs3d)
            return "((b * " + seq + " + " + t + ") * " + std::to_string(heads) + " + " + h + ") * " +
                   std::to_string(headSize);
         return "((b * " + std::to_string(heads) + " + " + h + ") * " + seq + " + " + t + ") * " +
                std::to_string(headSize);
      };

      std::stringstream out;
      out << "//---- Attention operator " << opName << "\n";
      out << SP << "{\n";
      out << SP << "std::vector<float> " << opName << "_score(" << kvSeq << ");\n";
      out << SP << "for (size_t b = 0; b < " << batch << "; b++) {\n";
      out << SP << "for (size_t h = 0; h < " << fQHeads << "; h++) {\n";
      out << SP << SP << "const size_t kvh = h / " << groupSize << ";\n";
      out << SP << SP << "for (size_t i = 0; i < " << qSeq << "; i++) {\n";
      out << SP << SP << SP << "const size_t qOffset = " << offset("h", "i", qSeq, fQHeads, fHeadSize) << ";\n";
      out << SP << SP << SP << "float maxScore = -std::numeric_limits<float>::infinity();\n";
      out << SP << SP << SP << "for (size_t j = 0; j < " << kvSeq << "; j++) {\n";
      out << SP << SP << SP << SP << "const size_t kOffset = " << offset("kvh", "j", kvSeq, fKVHeads, fHeadSize)
          << ";\n";
      out << SP << SP << SP << SP << "float s = 0;\n";
      out << SP << SP << SP << SP << "for (size_t d = 0; d < " << fHeadSize << "; d++)\n";
      out << SP << SP << SP << SP << SP << "s += tensor_" << fNQ << "[qOffset + d] * tensor_" << fNK
          << "[kOffset + d];\n";
      out << SP << SP << SP << SP << "s *= float(" << ConvertValToString(fScale) << ");\n";
      if (!fNMask.empty()) {
         // index into the mask, skipping its broadcast axes
         const std::vector<std::string> loopIdx = {"b", "h", "i", "j"};
         auto maskStrides = UTILITY::ComputeStrideFromShape(fShapeMask);
         std::string maskIndex;
         for (size_t a = 0; a < 4; a++) {
            if (fShapeMask[a].isParam || fShapeMask[a].dim != 1) {
               if (!maskIndex.empty())
                  maskIndex += " + ";
               maskIndex += loopIdx[a];
               if (a < 3)
                  maskIndex += " * " + maskStrides[a].GetVal();
            }
         }
         if (maskIndex.empty())
            maskIndex = "0";
         const std::string maskLen = fShapeMask[3].GetVal();
         out << SP << SP << SP << SP << "if (j >= " << maskLen << ")\n";
         out << SP << SP << SP << SP << SP << "s = -std::numeric_limits<float>::infinity();\n";
         if (fMaskIsBool) {
            out << SP << SP << SP << SP << "else if (!tensor_" << fNMask << "[" << maskIndex << "])\n";
            out << SP << SP << SP << SP << SP << "s = -std::numeric_limits<float>::infinity();\n";
         } else {
            out << SP << SP << SP << SP << "else\n";
            out << SP << SP << SP << SP << SP << "s += tensor_" << fNMask << "[" << maskIndex << "];\n";
         }
      }
      if (fIsCausal) {
         out << SP << SP << SP << SP << "if (j > i)\n";
         out << SP << SP << SP << SP << SP << "s = -std::numeric_limits<float>::infinity();\n";
      }
      if (fSoftcap > 0) {
         // as in the ONNX definition, the softcap is applied after adding the mask
         out << SP << SP << SP << SP << "s = float(" << ConvertValToString(fSoftcap) << ") * std::tanh(s / float("
             << ConvertValToString(fSoftcap) << "));\n";
      }
      out << SP << SP << SP << SP << opName << "_score[j] = s;\n";
      out << SP << SP << SP << SP << "maxScore = std::max(maxScore, s);\n";
      out << SP << SP << SP << "}\n";
      // softmax over the keys
      out << SP << SP << SP << "float sum = 0;\n";
      out << SP << SP << SP << "for (size_t j = 0; j < " << kvSeq << "; j++) {\n";
      out << SP << SP << SP << SP << opName << "_score[j] = std::exp(" << opName << "_score[j] - maxScore);\n";
      out << SP << SP << SP << SP << "sum += " << opName << "_score[j];\n";
      out << SP << SP << SP << "}\n";
      // Y = softmax * V
      out << SP << SP << SP << "const size_t yOffset = " << offset("h", "i", qSeq, fQHeads, fVHeadSize) << ";\n";
      out << SP << SP << SP << "for (size_t d = 0; d < " << fVHeadSize << "; d++) {\n";
      out << SP << SP << SP << SP << "float acc = 0;\n";
      out << SP << SP << SP << SP << "for (size_t j = 0; j < " << kvSeq << "; j++)\n";
      out << SP << SP << SP << SP << SP << "acc += " << opName << "_score[j] * tensor_" << fNV << "["
          << offset("kvh", "j", kvSeq, fKVHeads, fVHeadSize) << " + d];\n";
      out << SP << SP << SP << SP << "tensor_" << fNY << "[yOffset + d] = acc / sum;\n";
      out << SP << SP << SP << "}\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";
      out << SP << "}\n";
      out << SP << "}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override
   {
      return {std::string("algorithm"), std::string("cmath"), std::string("limits")};
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
