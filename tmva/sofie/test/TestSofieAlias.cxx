// Unit tests for the memory shared between an operator that only reinterprets
// its input and that input.
//
// The assertions are on the decision RModel makes, reported by IsAliasTensor(),
// not on the code it writes. EmittedCodeFollowsDecision is the exception: it
// checks that the operator acted on that answer, matching with a regex that
// ignores the spacing.

#include <TMVA/RModel.hxx>
#include <TMVA/RModelParser_ONNX.hxx>

#include <gtest/gtest.h>

#include <regex>
#include <string>
#include <vector>

#ifndef SOFIE_ONNX_MODELS_DIR
#define SOFIE_ONNX_MODELS_DIR "."
#endif

using namespace TMVA::Experimental::SOFIE;

namespace {

/// One model of generate_input_models.py, parsed and generated at `level`.
RModel generate(std::string const &modelName, OptimizationLevel level)
{
   RModelParser_ONNX parser;
   RModel model = parser.Parse(std::string(SOFIE_ONNX_MODELS_DIR) + "/" + modelName + ".onnx");
   model.SetOptimizationLevel(level);
   model.Generate(Options::kNoWeightFile);
   return model;
}

/// The output of one shape-only operator: `tensor` is produced by reading `owner`.
struct AliasCase {
   const char *model;
   const char *tensor;
   const char *owner;
   bool granted; ///< whether the alias is expected at OptimizationLevel::kExtended
   const char *why;
};

const std::vector<AliasCase> aliasCases = {
   {"ReshapeAlias", "reshaped", "prod", true, "a reshape only reinterprets the shape of its input"},
   {"SliceIdentityAlias", "sliced", "prod", true, "a slice selecting everything does not change the data"},
   {"IdentityAlias", "ident", "prod", true, "an identity is the same data under another name"},
   {"ReshapeAliasGraphOutput", "out", "prod", false, "a graph output is written into the buffer of the caller"},
   // the cases where the memory shared with the alias must outlive the alias itself
   {"AliasAcrossNewTensor", "reshaped", "prod", true, "the alias is read after another tensor was created"},
   {"AliasOwnerReadAfterAlias", "reshaped", "prod", true, "the aliased tensor is read after the last use of the alias"},
   {"AliasDynShape", "ident", "prod", true, "the aliased tensor comes from the dynamic memory pool"},
   // one row per link of the chain, each alias referring to the previous one
   {"AliasChain", "squeezed", "prod", true, "Squeeze does not change the data"},
   {"AliasChain", "unsqueezed", "squeezed", true, "Unsqueeze does not change the data"},
   {"AliasChain", "flattened", "unsqueezed", true, "Flatten does not change the data"},
   {"AliasChain", "reshaped", "flattened", true, "Reshape does not change the data"},
   {"AliasChainSingle", "reshaped", "prod", true, "the single alias the chain above amounts to"},
};

/// The declaration that points `tensor` at the memory of `owner`.
std::regex aliasRegex(AliasCase const &c)
{
   return std::regex(std::string("auto\\s*\\*\\s*tensor_") + c.tensor + "\\s*=\\s*tensor_" + c.owner + "\\s*;");
}

/// The copy of `owner` into the buffer of `tensor`, emitted where there is no alias.
std::regex copyRegex(AliasCase const &c)
{
   return std::regex(std::string("std::copy\\s*\\(\\s*tensor_") + c.owner + "\\b[^;]*\\btensor_" + c.tensor +
                     "\\s*\\)");
}

bool matches(std::string const &code, std::regex const &pattern)
{
   return std::regex_search(code, pattern);
}

} // namespace

/// An alias is granted exactly where the tensor it refers to has memory to share.
TEST(SofieAlias, GrantedAtExtended)
{
   for (auto const &c : aliasCases) {
      SCOPED_TRACE(std::string(c.model) + ": " + c.tensor);
      RModel model = generate(c.model, OptimizationLevel::kExtended);
      EXPECT_EQ(model.IsAliasTensor(c.tensor), c.granted) << c.why;
   }
}

/// At kBasic every tensor keeps a buffer of its own, as Clad requires.
TEST(SofieAlias, RefusedAtBasic)
{
   for (auto const &c : aliasCases) {
      SCOPED_TRACE(std::string(c.model) + ": " + c.tensor);
      RModel model = generate(c.model, OptimizationLevel::kBasic);
      EXPECT_FALSE(model.IsAliasTensor(c.tensor));
   }
}

/// The operator emits the pointer where the alias is granted and the copy where
/// it is refused.
TEST(SofieAlias, EmittedCodeFollowsDecision)
{
   for (auto const &c : aliasCases) {
      SCOPED_TRACE(std::string(c.model) + ": " + c.tensor);
      const std::regex alias = aliasRegex(c);
      const std::regex copy = copyRegex(c);

      const std::string extended = generate(c.model, OptimizationLevel::kExtended).ReturnGenerated();
      EXPECT_EQ(matches(extended, alias), c.granted);
      EXPECT_EQ(matches(extended, copy), !c.granted);

      const std::string basic = generate(c.model, OptimizationLevel::kBasic).ReturnGenerated();
      EXPECT_FALSE(matches(basic, alias));
      EXPECT_TRUE(matches(basic, copy));
   }
}

/// A chain of aliases costs no more memory than the single alias it amounts to.
/// Every link has to resolve to the tensor that owns the memory: a link pointing
/// at the previous alias instead leaves that memory reserved to the end, since
/// the tensor it names owns no chunk to release.
TEST(SofieAlias, ChainCostsNoExtraMemory)
{
   const RModel chain = generate("AliasChain", OptimizationLevel::kExtended);
   const RModel single = generate("AliasChainSingle", OptimizationLevel::kExtended);
   EXPECT_EQ(chain.GetIntermediateTensorSize(), single.GetIntermediateTensorSize());
}
