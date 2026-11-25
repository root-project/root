// TMVA SOFIE — minimal, dependency-free ONNX protobuf reader.
//
// Drop-in replacement for the protoc-generated onnx_proto3.pb.h. Provides the
// subset of the onnx:: message API that the SOFIE ONNX parser actually uses,
// backed by a hand-written protobuf wire-format decoder. No libprotobuf, no
// protoc, no codegen.
//
// Only the read-side accessors used by RModelParser_ONNX and the Parse*.cxx
// operator parsers are implemented.

#ifndef TMVA_SOFIE_ONNX_LITE
#define TMVA_SOFIE_ONNX_LITE

#include <cstdint>
#include <cstring>
#include <istream>
#include <memory>
#include <utility>
#include <sstream>
#include <string>
#include <vector>

// The messages live in TMVA::Experimental::SOFIE::onnx, not in the global onnx
// namespace the protoc-generated headers use. The real ONNX C++ library ships
// inside the `onnx` Python wheel, and when that extension module is loaded in
// the same process (as any PyTorch ONNX export does) its exported symbols
// interpose ours on ELF platforms: SOFIE would then call protobuf's
// onnx::TensorProto destructor on an object with this file's layout.
// Unqualified `onnx::` inside namespace SOFIE still names these classes, so
// the parser sources need no change.
namespace TMVA {
namespace Experimental {
namespace SOFIE {

namespace onnx {

// ---------------------------------------------------------------------------
// Protobuf wire-format reader (proto3 subset: varint, 64-bit, len, 32-bit)
// ---------------------------------------------------------------------------
namespace detail {

enum WireType {
   WT_VARINT = 0,
   WT_I64 = 1,
   WT_LEN = 2,
   WT_I32 = 5
};

class WireReader {
   const uint8_t *fP;
   const uint8_t *fEnd;
   bool fOk = true;

public:
   WireReader(const char *data, std::size_t n) : fP(reinterpret_cast<const uint8_t *>(data)), fEnd(fP + n) {}

   bool ok() const { return fOk; }
   bool eof() const { return fP >= fEnd; }

   uint64_t ReadVarint()
   {
      uint64_t result = 0;
      int shift = 0;
      while (fP < fEnd && shift < 64) {
         uint8_t b = *fP++;
         result |= uint64_t(b & 0x7F) << shift;
         if (!(b & 0x80))
            return result;
         shift += 7;
      }
      fOk = false;
      return result;
   }

   // Fixed-width fields are little-endian on the wire. Assemble them byte-wise
   // so the result is a correct host-order value on both little- and big-endian
   // machines (ROOT CI covers big-endian s390x). Varints need no such handling.
   uint32_t ReadFixed32()
   {
      if (fP + 4 > fEnd) {
         fOk = false;
         return 0;
      }
      uint32_t v = uint32_t(fP[0]) | (uint32_t(fP[1]) << 8) | (uint32_t(fP[2]) << 16) | (uint32_t(fP[3]) << 24);
      fP += 4;
      return v;
   }

   uint64_t ReadFixed64()
   {
      if (fP + 8 > fEnd) {
         fOk = false;
         return 0;
      }
      uint64_t v = 0;
      for (int k = 0; k < 8; ++k)
         v |= uint64_t(fP[k]) << (8 * k);
      fP += 8;
      return v;
   }

   // Length-delimited payload returned as a (ptr,len) view into the buffer.
   std::pair<const char *, std::size_t> ReadLen()
   {
      uint64_t n = ReadVarint();
      if (fP + n > fEnd) {
         fOk = false;
         return {nullptr, 0};
      }
      auto ptr = reinterpret_cast<const char *>(fP);
      fP += n;
      return {ptr, std::size_t(n)};
   }

   bool ReadTag(uint32_t &field, uint32_t &wire)
   {
      if (eof())
         return false;
      uint64_t tag = ReadVarint();
      if (!fOk)
         return false;
      field = uint32_t(tag >> 3);
      wire = uint32_t(tag & 7);
      return true;
   }

   void SkipField(uint32_t wire)
   {
      switch (wire) {
      case WT_VARINT: ReadVarint(); break;
      case WT_I64: ReadFixed64(); break;
      case WT_LEN: ReadLen(); break;
      case WT_I32: ReadFixed32(); break;
      default: fOk = false; break; // groups (3,4) unsupported / not used by ONNX
      }
   }
};

// Read a repeated numeric field that may be packed (single WT_LEN block) or
// written as individual entries.
inline void ReadPackedVarint(WireReader &r, uint32_t wire, std::vector<int64_t> &out)
{
   if (wire == WT_LEN) {
      auto s = r.ReadLen();
      WireReader rr(s.first, s.second);
      while (!rr.eof())
         out.push_back(int64_t(rr.ReadVarint()));
   } else {
      out.push_back(int64_t(r.ReadVarint()));
   }
}
inline void ReadPackedI32(WireReader &r, uint32_t wire, std::vector<int32_t> &out)
{
   if (wire == WT_LEN) {
      auto s = r.ReadLen();
      WireReader rr(s.first, s.second);
      while (!rr.eof())
         out.push_back(int32_t(rr.ReadVarint()));
   } else {
      out.push_back(int32_t(r.ReadVarint()));
   }
}
inline void ReadPackedFloat(WireReader &r, uint32_t wire, std::vector<float> &out)
{
   auto one = [](uint32_t bits) {
      float f;
      std::memcpy(&f, &bits, 4);
      return f;
   };
   if (wire == WT_LEN) {
      auto s = r.ReadLen();
      WireReader rr(s.first, s.second);
      while (!rr.eof())
         out.push_back(one(rr.ReadFixed32()));
   } else {
      out.push_back(one(r.ReadFixed32()));
   }
}
inline void ReadPackedDouble(WireReader &r, uint32_t wire, std::vector<double> &out)
{
   auto one = [](uint64_t bits) {
      double d;
      std::memcpy(&d, &bits, 8);
      return d;
   };
   if (wire == WT_LEN) {
      auto s = r.ReadLen();
      WireReader rr(s.first, s.second);
      while (!rr.eof())
         out.push_back(one(rr.ReadFixed64()));
   } else {
      out.push_back(one(r.ReadFixed64()));
   }
}

inline std::string Str(std::pair<const char *, std::size_t> v)
{
   return std::string(v.first, v.second);
}

} // namespace detail

// ---------------------------------------------------------------------------
// Message types (subset). Each exposes the generated-protobuf-style accessors.
// ---------------------------------------------------------------------------

class GraphProto;  // fwd
class TensorProto; // fwd

// --- TensorShapeProto::Dimension (flattened name: TensorShapeProto_Dimension)
class TensorShapeProto_Dimension {
public:
   enum class ValueCase {
      VALUE_NOT_SET = 0,
      kDimValue = 1,
      kDimParam = 2
   };

   ValueCase value_case() const { return fCase; }
   int64_t dim_value() const { return fDimValue; }
   const std::string &dim_param() const { return fDimParam; }

   void ParseFrom(detail::WireReader r)
   {
      uint32_t f, wt;
      while (r.ReadTag(f, wt)) {
         switch (f) {
         case 1:
            fDimValue = int64_t(r.ReadVarint());
            fCase = ValueCase::kDimValue;
            break;
         case 2:
            fDimParam = detail::Str(r.ReadLen());
            fCase = ValueCase::kDimParam;
            break;
         default: r.SkipField(wt); break;
         }
      }
   }

private:
   ValueCase fCase = ValueCase::VALUE_NOT_SET;
   int64_t fDimValue = 0;
   std::string fDimParam;
};

class TensorShapeProto {
public:
   int dim_size() const { return int(fDim.size()); }
   const TensorShapeProto_Dimension &dim(int i) const { return fDim[i]; }

   void ParseFrom(detail::WireReader r)
   {
      uint32_t f, wt;
      while (r.ReadTag(f, wt)) {
         if (f == 1 && wt == detail::WT_LEN) {
            auto s = r.ReadLen();
            fDim.emplace_back();
            fDim.back().ParseFrom(detail::WireReader(s.first, s.second));
         } else {
            r.SkipField(wt);
         }
      }
   }

private:
   std::vector<TensorShapeProto_Dimension> fDim;
};

// TypeProto::Tensor
class TypeProto_Tensor {
public:
   int elem_type() const { return fElemType; }
   bool has_shape() const { return fHasShape; }
   const TensorShapeProto &shape() const { return fShape; }

   void ParseFrom(detail::WireReader r)
   {
      uint32_t f, wt;
      while (r.ReadTag(f, wt)) {
         switch (f) {
         case 1: fElemType = int(r.ReadVarint()); break;
         case 2: {
            auto s = r.ReadLen();
            fShape.ParseFrom(detail::WireReader(s.first, s.second));
            fHasShape = true;
            break;
         }
         default: r.SkipField(wt); break;
         }
      }
   }

private:
   int fElemType = 0;
   bool fHasShape = false;
   TensorShapeProto fShape;
};

class TypeProto {
public:
   const TypeProto_Tensor &tensor_type() const { return fTensorType; }

   void ParseFrom(detail::WireReader r)
   {
      uint32_t f, wt;
      while (r.ReadTag(f, wt)) {
         if (f == 1 && wt == detail::WT_LEN) { // tensor_type
            auto s = r.ReadLen();
            fTensorType.ParseFrom(detail::WireReader(s.first, s.second));
         } else {
            r.SkipField(wt);
         }
      }
   }

private:
   TypeProto_Tensor fTensorType;
};

class ValueInfoProto {
public:
   const std::string &name() const { return fName; }
   const TypeProto &type() const { return fType; }

   void ParseFrom(detail::WireReader r)
   {
      uint32_t f, wt;
      while (r.ReadTag(f, wt)) {
         switch (f) {
         case 1: fName = detail::Str(r.ReadLen()); break;
         case 2: {
            auto s = r.ReadLen();
            fType.ParseFrom(detail::WireReader(s.first, s.second));
            break;
         }
         default: r.SkipField(wt); break;
         }
      }
   }

private:
   std::string fName;
   TypeProto fType;
};

class StringStringEntryProto {
public:
   const std::string &key() const { return fKey; }
   const std::string &value() const { return fValue; }

   void ParseFrom(detail::WireReader r)
   {
      uint32_t f, wt;
      while (r.ReadTag(f, wt)) {
         switch (f) {
         case 1: fKey = detail::Str(r.ReadLen()); break;
         case 2: fValue = detail::Str(r.ReadLen()); break;
         default: r.SkipField(wt); break;
         }
      }
   }

private:
   std::string fKey, fValue;
};

class TensorProto {
public:
   enum DataType {
      UNDEFINED = 0,
      FLOAT = 1,
      UINT8 = 2,
      INT8 = 3,
      UINT16 = 4,
      INT16 = 5,
      INT32 = 6,
      INT64 = 7,
      STRING = 8,
      BOOL = 9,
      FLOAT16 = 10,
      DOUBLE = 11,
      UINT32 = 12,
      UINT64 = 13,
      COMPLEX64 = 14,
      COMPLEX128 = 15,
      BFLOAT16 = 16
   };
   enum DataLocation {
      DEFAULT = 0,
      EXTERNAL = 1
   };

   const std::string &name() const { return fName; }
   int data_type() const { return fDataType; }
   int dims_size() const { return int(fDims.size()); }
   int64_t dims(int i) const { return fDims[i]; }
   const std::string &raw_data() const { return fRawData; }
   DataLocation data_location() const { return fDataLocation; }
   const std::vector<StringStringEntryProto> &external_data() const { return fExternalData; }

   int float_data_size() const { return int(fFloatData.size()); }
   int double_data_size() const { return int(fDoubleData.size()); }
   int int32_data_size() const { return int(fInt32Data.size()); }
   int int64_data_size() const { return int(fInt64Data.size()); }
   const std::vector<float> &float_data() const { return fFloatData; }
   const std::vector<double> &double_data() const { return fDoubleData; }
   const std::vector<int32_t> &int32_data() const { return fInt32Data; }
   const std::vector<int64_t> &int64_data() const { return fInt64Data; }
   // Indexed element accessors (the generated protobuf API exposes both forms).
   float float_data(int i) const { return fFloatData[i]; }
   double double_data(int i) const { return fDoubleData[i]; }
   int32_t int32_data(int i) const { return fInt32Data[i]; }
   int64_t int64_data(int i) const { return fInt64Data[i]; }

   void ParseFrom(detail::WireReader r)
   {
      uint32_t f, wt;
      while (r.ReadTag(f, wt)) {
         switch (f) {
         case 1: detail::ReadPackedVarint(r, wt, fDims); break;
         case 2: fDataType = int(r.ReadVarint()); break;
         case 4: detail::ReadPackedFloat(r, wt, fFloatData); break;
         case 5: detail::ReadPackedI32(r, wt, fInt32Data); break;
         case 7: detail::ReadPackedVarint(r, wt, fInt64Data); break;
         case 8: fName = detail::Str(r.ReadLen()); break;
         case 9: fRawData = detail::Str(r.ReadLen()); break;
         case 10: detail::ReadPackedDouble(r, wt, fDoubleData); break;
         case 13: {
            auto s = r.ReadLen();
            fExternalData.emplace_back();
            fExternalData.back().ParseFrom(detail::WireReader(s.first, s.second));
            break;
         }
         case 14: fDataLocation = DataLocation(r.ReadVarint()); break;
         default: r.SkipField(wt); break;
         }
      }
   }

private:
   std::string fName;
   int fDataType = 0;
   std::vector<int64_t> fDims;
   std::string fRawData;
   DataLocation fDataLocation = DEFAULT;
   std::vector<StringStringEntryProto> fExternalData;
   std::vector<float> fFloatData;
   std::vector<double> fDoubleData;
   std::vector<int32_t> fInt32Data;
   std::vector<int64_t> fInt64Data;
};

class AttributeProto {
public:
   enum AttributeType {
      UNDEFINED = 0,
      FLOAT = 1,
      INT = 2,
      STRING = 3,
      TENSOR = 4,
      GRAPH = 5,
      FLOATS = 6,
      INTS = 7,
      STRINGS = 8,
      TENSORS = 9,
      GRAPHS = 10,
      SPARSE_TENSOR = 11,
      SPARSE_TENSORS = 12
   };

   const std::string &name() const { return fName; }
   AttributeType type() const { return fType; }
   float f() const { return fF; }
   int64_t i() const { return fI; }
   const std::string &s() const { return fS; }
   const TensorProto &t() const { return fT; }
   const GraphProto &g() const; // defined after GraphProto
   bool has_g() const { return fHasG; }
   bool has_t() const { return fHasT; }

   const std::vector<float> &floats() const { return fFloats; }
   const std::vector<int64_t> &ints() const { return fInts; }
   const std::vector<std::string> &strings() const { return fStrings; }

   void ParseFrom(detail::WireReader r); // defined after GraphProto (needs g())

private:
   std::string fName;
   AttributeType fType = UNDEFINED;
   float fF = 0.f;
   int64_t fI = 0;
   std::string fS;
   bool fHasG = false, fHasT = false;
   std::vector<float> fFloats;
   std::vector<int64_t> fInts;
   std::vector<std::string> fStrings;
   TensorProto fT;
   // graph attribute held by shared_ptr to break the cyclic type dependency
   // (a subgraph contains nodes, whose attributes may again hold subgraphs)
   std::shared_ptr<GraphProto> fG;
};

class NodeProto {
public:
   const std::string &op_type() const { return fOpType; }
   const std::string &name() const { return fName; }
   int input_size() const { return int(fInput.size()); }
   const std::string &input(int i) const { return fInput[i]; }
   const std::vector<std::string> &input() const { return fInput; }
   int output_size() const { return int(fOutput.size()); }
   const std::string &output(int i) const { return fOutput[i]; }
   const std::vector<std::string> &output() const { return fOutput; }
   int attribute_size() const { return int(fAttribute.size()); }
   const AttributeProto &attribute(int i) const { return fAttribute[i]; }

   void ParseFrom(detail::WireReader r)
   {
      uint32_t f, wt;
      while (r.ReadTag(f, wt)) {
         switch (f) {
         case 1: fInput.push_back(detail::Str(r.ReadLen())); break;
         case 2: fOutput.push_back(detail::Str(r.ReadLen())); break;
         case 3: fName = detail::Str(r.ReadLen()); break;
         case 4: fOpType = detail::Str(r.ReadLen()); break;
         case 5: {
            auto s = r.ReadLen();
            fAttribute.emplace_back();
            fAttribute.back().ParseFrom(detail::WireReader(s.first, s.second));
            break;
         }
         default: r.SkipField(wt); break;
         }
      }
   }

private:
   std::vector<std::string> fInput, fOutput;
   std::string fName, fOpType;
   std::vector<AttributeProto> fAttribute;
};

class GraphProto {
public:
   const std::string &name() const { return fName; }
   int node_size() const { return int(fNode.size()); }
   const NodeProto &node(int i) const { return fNode[i]; }
   int input_size() const { return int(fInput.size()); }
   const ValueInfoProto &input(int i) const { return fInput[i]; }
   int output_size() const { return int(fOutput.size()); }
   const ValueInfoProto &output(int i) const { return fOutput[i]; }
   int initializer_size() const { return int(fInitializer.size()); }
   const TensorProto &initializer(int i) const { return fInitializer[i]; }

   void ParseFrom(detail::WireReader r)
   {
      uint32_t f, wt;
      while (r.ReadTag(f, wt)) {
         switch (f) {
         case 1: {
            auto s = r.ReadLen();
            fNode.emplace_back();
            fNode.back().ParseFrom(detail::WireReader(s.first, s.second));
            break;
         }
         case 2: fName = detail::Str(r.ReadLen()); break;
         case 5: {
            auto s = r.ReadLen();
            fInitializer.emplace_back();
            fInitializer.back().ParseFrom(detail::WireReader(s.first, s.second));
            break;
         }
         case 11: {
            auto s = r.ReadLen();
            fInput.emplace_back();
            fInput.back().ParseFrom(detail::WireReader(s.first, s.second));
            break;
         }
         case 12: {
            auto s = r.ReadLen();
            fOutput.emplace_back();
            fOutput.back().ParseFrom(detail::WireReader(s.first, s.second));
            break;
         }
         default: r.SkipField(wt); break;
         }
      }
   }

private:
   std::string fName;
   std::vector<NodeProto> fNode;
   std::vector<TensorProto> fInitializer;
   std::vector<ValueInfoProto> fInput, fOutput;
};

// --- AttributeProto members that depend on the complete GraphProto type ------
inline const GraphProto &AttributeProto::g() const
{
   return *fG;
}

inline void AttributeProto::ParseFrom(detail::WireReader r)
{
   uint32_t f, wt;
   while (r.ReadTag(f, wt)) {
      switch (f) {
      case 1: fName = detail::Str(r.ReadLen()); break;
      case 2: {
         uint32_t bits = r.ReadFixed32();
         std::memcpy(&fF, &bits, 4);
         break;
      }
      case 3: fI = int64_t(r.ReadVarint()); break;
      case 4: fS = detail::Str(r.ReadLen()); break;
      case 5: {
         auto s = r.ReadLen();
         fT.ParseFrom(detail::WireReader(s.first, s.second));
         fHasT = true;
         break;
      }
      case 6: {
         auto s = r.ReadLen();
         fG = std::make_shared<GraphProto>();
         fG->ParseFrom(detail::WireReader(s.first, s.second));
         fHasG = true;
         break;
      }
      case 7: detail::ReadPackedFloat(r, wt, fFloats); break;
      case 8: detail::ReadPackedVarint(r, wt, fInts); break;
      case 9: fStrings.push_back(detail::Str(r.ReadLen())); break;
      case 20: fType = AttributeType(r.ReadVarint()); break;
      default: r.SkipField(wt); break;
      }
   }
}

class ModelProto {
public:
   int64_t ir_version() const { return fIrVersion; }
   const std::string &producer_name() const { return fProducerName; }
   const GraphProto &graph() const { return fGraph; }

   bool ParseFromIstream(std::istream *in)
   {
      std::ostringstream ss;
      ss << in->rdbuf();
      fBuffer = ss.str();
      detail::WireReader r(fBuffer.data(), fBuffer.size());
      uint32_t f, wt;
      while (r.ReadTag(f, wt)) {
         switch (f) {
         case 1: fIrVersion = int64_t(r.ReadVarint()); break;
         case 2: fProducerName = detail::Str(r.ReadLen()); break;
         case 7: {
            auto s = r.ReadLen();
            fGraph.ParseFrom(detail::WireReader(s.first, s.second));
            break;
         }
         default: r.SkipField(wt); break;
         }
      }
      return r.ok();
   }

private:
   std::string fBuffer;
   int64_t fIrVersion = 0;
   std::string fProducerName;
   GraphProto fGraph;
};

} // namespace onnx

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
