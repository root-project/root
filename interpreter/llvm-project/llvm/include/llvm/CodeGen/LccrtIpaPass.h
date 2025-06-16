#ifndef LLVM_ANALYSIS_COLLECT_AA_RESULTS_FOR_LCC_PASS_H
#define LLVM_ANALYSIS_COLLECT_AA_RESULTS_FOR_LCC_PASS_H

#include <set>
#include <unordered_map>

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

namespace llvm {

using Values = std::vector<MemoryLocation>;
using AliasVector = std::vector<AliasResult>;
using Aliases = std::vector<AliasVector>;

struct FunctionIPAResults {
    Function &function;
    Values values;
    Aliases aliases;

    FunctionIPAResults() = default;
    FunctionIPAResults(Function &);

    void
    addValue(const MemoryLocation &);

    size_t
    length() const;

    bool
    isEmpty() const;

    Values::const_iterator
    begin() const;

    Values::const_iterator
    end() const;

    Values::iterator
    begin();

    Values::iterator
    end();

    template <typename T>
    void collectAliasInfo(T&, SimpleAAQueryInfo&);

    void *
    getAliases() const;
};


using IPAResults = std::unordered_map<GlobalValue::GUID, FunctionIPAResults>;


class LccrtIpaPass : public ModulePass {
public:
    static char ID;

    LccrtIpaPass();

    bool runOnModule(Module &) override;
    void getAnalysisUsage(AnalysisUsage &) const override;

    IPAResults *getIPAResults();

private:
    void pickValues(Module &);
    void pickValuesFromFunction(Function &);

    IPAResults ipaResults;
};

ModulePass *createLccrtIpaPass();

} // llvm

#endif
