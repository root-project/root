#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_BASIC_UNARY
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_BASIC_UNARY

#include <TMVA/ROperator.hxx>
#include <TMVA/RModel.hxx>
#include <TMVA/SOFIE_common.hxx>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

enum class EBasicUnaryOperator { kReciprocal, kSqrt , kNeg, kExp, kLog, kSin, kCos };

template <typename T, EBasicUnaryOperator Op>
struct UnaryOpTraits {
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kReciprocal> {
   static std::string Name() { return "Reciprocal"; }
   static std::string Op(const std::string &X) { return "1/" + X; }
   static std::string Op_GPU(const std::string &X) { return "cl::sycl::native::recip(" + X + ")";}
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kSqrt> {
   static std::string Name() { return "Sqrt"; }
   static std::string Op(const std::string &X) { return "std::sqrt(" + X + ")"; }
   static std::string Op_GPU(const std::string &X) { return "cl::sycl::sqrt(" + X + ")";}
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kNeg> {
   static std::string Name() { return "Neg"; }
   static std::string Op(const std::string &X) { return "-" + X; }
   static std::string Op_GPU(const std::string &X) { return "-" + X;}
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kExp> {
   static std::string Name() { return "Exp"; }
   static std::string Op(const std::string &X) { return "std::exp(" + X + ")"; }
   static std::string Op_GPU(const std::string &X) { return "cl::sycl::exp(" + X + ")";}
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kLog> {
   static std::string Name() { return "Log"; }
   static std::string Op(const std::string &X) { return "std::log(" + X + ")"; }
   static std::string Op_GPU(const std::string &X) { return "cl::sycl::native::recip(" + X + ")";}
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kSin> {
   static std::string Name() { return "Sin"; }
   static std::string Op(const std::string &X) { return "std::sin(" + X + ")"; }
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kCos> {
   static std::string Name() { return "Cos"; }
   static std::string Op(const std::string &X) { return "std::cos(" + X + ")"; }
};

template <typename T, EBasicUnaryOperator Op>
class ROperator_BasicUnary final : public ROperator {
private:
   std::string fNX;
   std::string fNY;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeY;

public:
   ROperator_BasicUnary() {}

   ROperator_BasicUnary(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {}

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override { return input; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override { return input; }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA::SOFIE - Tensor " + fNX + " not found.");
      }
      fShapeX = model.GetTensorShape(fNX);
      fShapeY = ShapeInference({fShapeX})[0];
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      std::stringstream out;

      out << SP << "\n//---- Operator" << UnaryOpTraits<T, Op>::Name() << " " << OpName << "\n";
      size_t length = ConvertShapeToLength(fShapeX);
      out << SP << "for (size_t i = 0; i < " << length << "; i++) {\n";
      out << SP << SP << "tensor_" << fNY << "[i] = " << UnaryOpTraits<T, Op>::Op("tensor_" + fNX + "[i]") << ";\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string GenerateGPU(std::string OpName, std::string gemm, std::string copy, 
   std::string axpy, std::string transpose, std::string nontrans, std::string trans, std::string copy_batch, std::string scal) override {
      OpName = "op_" + OpName;
      std::stringstream out;
      out << "\n" << SP*3 << "//---- Operator" << UnaryOpTraits<T, Op>::Name() << " " << OpName << "\n";
      size_t length = ConvertShapeToLength(fShapeX);
      
      out << SP*3 << "q.submit([&](cl::sycl::handler& cgh) {\n";
      out << SP*4 << "auto acc_tensor_" << fNX << "= cl::sycl::accessor{buf_tensor_" << fNX;
      out << ", cgh, cl::sycl::read_only};\n";
      out << SP*4 << "auto acc_tensor_" << fNY << "= cl::sycl::accessor{buf_tensor_" << fNY;
      out << ", cgh, cl::sycl::write_only, cl::sycl::no_init};\n";
      out << SP*4 <<  "cgh.parallel_for<class " << OpName << ">(cl::sycl::range<1>(" << length;
      out << "), [=](cl::sycl::id<1> id){\n";
      out << SP*5 << "acc_tensor_" << fNY << "[id] = " << UnaryOpTraits<T, Op>::Op_GPU("acc_tensor_" + fNX + "[id]") << ";\n";
      out << SP*4 << "});\n";
      out << SP*3 << "});\n";

      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {
      if (Op == EBasicUnaryOperator::kSqrt || Op == EBasicUnaryOperator::kExp || Op == EBasicUnaryOperator::kLog) {
         return { std::string("cmath") };
      } else {
         return {};
      }
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
