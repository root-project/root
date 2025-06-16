#include <iostream>
#include <unordered_set>
#include <utility>

#include "llvm/CodeGen/LccrtIpaPass.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/FileSystem.h"

using namespace llvm;

char LccrtIpaPass::ID = 0;
INITIALIZE_PASS_BEGIN( LccrtIpaPass, "lccrt-ipa",
                       "Collect IPA Results for lccrt-backend", false, true)
INITIALIZE_PASS_END( LccrtIpaPass, "lccrt-ipa",
                     "Collect IPA Results for lccrt-backend", false, true)

FunctionIPAResults::FunctionIPAResults(Function &function) :
    function(function)
{}

void
FunctionIPAResults::addValue(const MemoryLocation &loc)
{
    values.push_back(loc);
}

size_t
FunctionIPAResults::length() const
{
    return values.size();
}

bool
FunctionIPAResults::isEmpty() const
{
    return values.empty();
}

Values::const_iterator
FunctionIPAResults::begin() const
{
    return values.begin();
}

Values::const_iterator
FunctionIPAResults::end() const
{
    return values.end();
}

Values::iterator
FunctionIPAResults::begin()
{
    return values.begin();
}

Values::iterator
FunctionIPAResults::end()
{
    return values.end();
}

template<typename T>
void
FunctionIPAResults::collectAliasInfo(T &aaResult, SimpleAAQueryInfo &AAQI)
{
    if (aliases.empty()) {
        aliases.resize(values.size());
        for (auto& i : aliases) {
            i.resize(values.size(), AliasResult::MayAlias);
        }
    }

    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = 0; j < values.size(); ++j) {
            if (aliases[i][j] == AliasResult::MayAlias) {
                aliases[i][j] = aaResult.alias(values[i], values[j], AAQI, nullptr);
            }
        }
    }
}

void *
FunctionIPAResults::getAliases() const
{
    auto size = aliases.size() * aliases.size();
    auto *result = (AliasResult *) malloc(sizeof(AliasResult *) * size);
    if (result) {
        for (size_t i = 0; i < aliases.size(); ++i) {
            memcpy(result + i * aliases.size(),
                   aliases[i].data(),
                   sizeof(AliasResult) * aliases[i].size());
        }
    }
    return result;
}


void
LccrtIpaPass::pickValues(Module& M)
{
    for (auto &f: M) {
        pickValuesFromFunction(f);
    }
}

void
LccrtIpaPass::pickValuesFromFunction(Function &F)
{
    if (F.empty() || F.isDeclaration()) {
        return;
    }

    auto it = ipaResults.insert(std::make_pair(F.getGUID(), FunctionIPAResults(F))).first;
    auto &r = it->second;

    for (auto& b : F) {
        for (auto& i : b) {
            if (isa<llvm::LoadInst>(i) || isa<llvm::StoreInst>(i)) {
                r.addValue(MemoryLocation::get(&i));
            }
        }
    }
}

LccrtIpaPass::LccrtIpaPass() :
    ModulePass(ID)
{
    //initializeLccrtIpaPassPass(*PassRegistry::getPassRegistry());
}

bool
LccrtIpaPass::runOnModule(Module &M)
{
    pickValues(M);

    if (!ipaResults.empty()) {
        for (auto &i : ipaResults) {
            auto &aaResults =
                getAnalysis<AAResultsWrapperPass>(i.second.function).getAAResults();

            SimpleAAQueryInfo AAQI(aaResults);

        auto tbaa = getAnalysisIfAvailable<TypeBasedAAWrapperPass>();
        if (tbaa) {
            auto &tbaaResults = tbaa->getResult();
                i.second.collectAliasInfo(tbaaResults, AAQI);
        } else {
            std::cerr << "No TBAA available\n";
        }

            i.second.collectAliasInfo(aaResults, AAQI);
        }
    }

    return false;
}

IPAResults *
LccrtIpaPass::getIPAResults()
{
    return &ipaResults;
}

void
LccrtIpaPass::getAnalysisUsage(AnalysisUsage &AU) const
{
    AU.setPreservesAll();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<TypeBasedAAWrapperPass>();
}

namespace llvm {

ModulePass *
createLccrtIpaPass()
{
    return new LccrtIpaPass();
}

} // llvm
