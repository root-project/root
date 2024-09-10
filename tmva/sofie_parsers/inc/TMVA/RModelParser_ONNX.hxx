#ifndef TMVA_SOFIE_RMODELPARSER_ONNX
#define TMVA_SOFIE_RMODELPARSER_ONNX

#include "TMVA/RModel.hxx"

#include <memory>
#include <functional>
#include <unordered_map>

// forward declaration
namespace onnx {
class NodeProto;
class GraphProto;
} // namespace onnx

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RModelParser_ONNX;

using ParserFuncSignature =
   std::function<std::unique_ptr<ROperator>(RModelParser_ONNX & /*parser*/, const onnx::NodeProto & /*nodeproto*/)>;
using ParserFuseFuncSignature =
   std::function<std::unique_ptr<ROperator> (RModelParser_ONNX& /*parser*/, const onnx::NodeProto& /*firstnode*/, const onnx::NodeProto& /*secondnode*/)>;

class RModelParser_ONNX {
public:
   struct OperatorsMapImpl;

private:

   bool fVerbose = false;
   // Registered operators
   std::unique_ptr<OperatorsMapImpl> fOperatorsMapImpl;
   // Type of the tensors
   std::unordered_map<std::string, ETensorType> fTensorTypeMap;

   // all model inputs
   std::map<std::string, int> allInputs;


public:
   // Register an ONNX operator
   void RegisterOperator(const std::string &name, ParserFuncSignature func);

   // Check if the operator is registered
   bool IsRegisteredOperator(const std::string &name);

   // List of registered operators
   std::vector<std::string> GetRegisteredOperators();

   // Set the type of the tensor
   void RegisterTensorType(const std::string & /*name*/, ETensorType /*type*/);

   // Check if the type of the tensor is registered
   bool IsRegisteredTensorType(const std::string & /*name*/);

   // check verbosity
   bool Verbose() const {
      return fVerbose;
   }

   // Get the type of the tensor
   ETensorType GetTensorType(const std::string &name);

   // Parse the index'th node from the ONNX graph
   std::unique_ptr<ROperator> ParseOperator(const size_t /*index*/, const onnx::GraphProto & /*graphproto*/,
                                            const std::vector<size_t> & /*nodes*/);

   // parse the ONNX graph
   void ParseONNXGraph(RModel & model, const onnx::GraphProto & g, std::string  name = "");

public:

   RModelParser_ONNX() noexcept;

   RModel Parse(std::string filename, bool verbose = false);


   ~RModelParser_ONNX();
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_RMODELPARSER_ONNX
