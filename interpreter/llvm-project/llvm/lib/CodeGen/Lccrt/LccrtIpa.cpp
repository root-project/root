//===-- LccrtIpa.cpp - Lccrt emittion ipa-results ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LccrtModuleIpaEmitter/LccrtFunctionIpaEmitter classes.
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instruction.h"
#include "LccrtIpa.h"

#ifdef LLVM_WITH_LCCRT
using namespace llvm;

MetaAliases *
wrapInArray(void *aliases, unsigned int length)
{
    auto result = (MetaAliases *) malloc(sizeof( MetaAliases));
    if (result) {
        result->items = aliases;
        result->length = length;
    }

    return result;
}

void
func_metadata_delete(lccrt_function_ptr f, uintptr_t value)
{
    auto *a = (MetaAliases *) value;
    auto *aliases = (AliasResult *) a->items;

    free(aliases);
    free(a);
}

void
oper_metadata_delete(lccrt_oper_ptr oper, uintptr_t value)
{
    auto id = (unsigned int *) value;

    free(id);
}

IPAResults *
LccrtModuleIpaEmitter::findIPAResults( Pass *pass)
{
    LccrtIpaPass *ipa_pass = pass->getAnalysisIfAvailable<LccrtIpaPass>();

    return (ipa_pass ? ipa_pass->getIPAResults() : 0);
} /* LccrtModuleIpaEmitter::findIPAResults */

LccrtModuleIpaEmitter::LccrtModuleIpaEmitter( Pass *parentPass) :
    parentPass( parentPass),
    ipaResults( findIPAResults( parentPass))
{
} /* LccrtModuleIpaEmitter::LccrtModuleIpaEmitter */

void
LccrtModuleIpaEmitter::open( lccrt_module_ptr m, const Module *)
{
    ecat_ipa = lccrt_module_new_einfo_category( m, "ipa");
} /* LccrtModuleIpaEmitter::open */

void
LccrtModuleIpaEmitter::close()
{
    return;
} /* LccrtModuleIpaEmitter::close */

lccrt_eic_t
LccrtModuleIpaEmitter::getMetadata() const
{
    return ecat_ipa;
}

const FunctionIPAResults *
LccrtModuleIpaEmitter::getFunctionIPAResults(GlobalValue::GUID functionGUID) const
{
    if (!ipaResults) {
        return nullptr;
    }

    auto found = ipaResults->find(functionGUID);

    return found != ipaResults->end() ?  &found->second : nullptr;
    }

LccrtFunctionIpaEmitter::LccrtFunctionIpaEmitter( LccrtModuleIpaEmitter *mipa,
                                                  lccrt_function_ptr f,
                                                  const Function *F) :
    mipa(mipa),
    functionIPAResults(nullptr),
    aliases(nullptr)
{
    lccrt_module_ptr m;
    lccrt_eir_t eref_fdata;
    void *a;
    uint64_t length;

    m = lccrt_function_get_module( f);

    if (mipa) {
        ecat_ipa = mipa->getMetadata();
        functionIPAResults = F->hasName() ?
            mipa->getFunctionIPAResults( F->getGUID()) :
            nullptr;
    if (functionIPAResults) {
        a = functionIPAResults->getAliases();
            if (a) {
                length = functionIPAResults->aliases.size();
                eref_fdata = lccrt_einfo_new_raw( m, sizeof(length), (uint8_t *) &length);
                lccrt_function_set_einfo( f, ecat_ipa, eref_fdata);
    }
    }
    }
} /* LccrtFunctionIpaEmitter::LccrtFunctionIpaEmitter */

void
LccrtFunctionIpaEmitter::setOperIpaResult( lccrt_oper_ptr oper, const Instruction &Inst)
{
    lccrt_module_ptr m;
    lccrt_function_ptr f;
    lccrt_eitd_ptr flds_types[3], eitd, eitd64, eitdarr;
    lccrt_eir_t eref_odata, eref_data, eref_elemdata;
    lccrt_eifi_t fid;
    const char *flds_names[] = { "length", "id", "aliases", };

    if (!lccrt_einfo_category_is_valued( ecat_ipa) || !functionIPAResults) {
        return;
    }

    if (!isa<LoadInst>(Inst) && !isa<StoreInst>(Inst)) {
        return;
    }

    f = lccrt_oper_get_function( oper);
    m = lccrt_function_get_module( f);

    eitd64 = lccrt_einfo_make_tydescr_i64( m);
    eitdarr = lccrt_einfo_make_tydescr_array( m, eitd64);

    flds_types[0] = eitd64;
    flds_types[1] = eitd64;
    flds_types[2] = eitdarr;

    auto length = functionIPAResults->length();
    for (size_t i = 0; i < length; ++i) {
        auto loc = MemoryLocation::get(&Inst);
        if (functionIPAResults->values[i] == loc) {
            eitd = lccrt_einfo_make_tydescr_struct( m, "ipa", 3, flds_names, flds_types);
            eref_odata = lccrt_einfo_new_struct( eitd);

            fid = lccrt_einfo_find_tydescr_field( eitd, "length");
            eref_data = lccrt_einfo_new_i64( length);
            lccrt_einfo_set_field( eref_odata, fid, eref_data);

            fid = lccrt_einfo_find_tydescr_field( eitd, "id");
            eref_data = lccrt_einfo_new_i64( i);
            lccrt_einfo_set_field( eref_odata, fid, eref_data);

            fid = lccrt_einfo_find_tydescr_field( eitd, "aliases");
            eref_data = lccrt_einfo_new_array( eitdarr, 0);
            lccrt_einfo_set_field( eref_odata, fid, eref_data);

            for (auto &j : functionIPAResults->aliases[i]) {
                eref_elemdata = lccrt_einfo_new_i64( j == AliasResult::NoAlias);
                lccrt_einfo_push_elem( eref_data, eref_elemdata);
            }

            lccrt_oper_set_einfo( oper, ecat_ipa, eref_odata);

                return;
            }
        }
} /* LccrtFunctionIpaEmitter::LccrtFunctionIpaEmitter */
#endif /* LLVM_WITH_LCCRT */
