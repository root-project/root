//=- llvm/CodeGen/Lccrt/LccrtDbg.h - Lccrt-IR translation -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LccrtDbg class, which
// implements translation dwarf-metadata from LLVM-IR to metadata of LCCRT-IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_LCCRT_LCCRTDBG_H
#define LLVM_LIB_CODEGEN_LCCRT_LCCRTDBG_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"

#ifdef LLVM_WITH_LCCRT
#include "lccrt.h"

#include <map>

namespace llvm {

class LccrtDbgEtydes {
public:
    struct ModuleData {
        lccrt_einfo_tydescr_ptr etyde;
        lccrt_einfo_field_id_t k_types;
        lccrt_einfo_field_id_t k_gvars;
    };
    struct Type {
        lccrt_einfo_tydescr_ptr etyde;
        lccrt_einfo_tydescr_ptr etyde_ai64; // массив из i64
        lccrt_einfo_field_id_t k_ident;
        lccrt_einfo_field_id_t k_type;
        lccrt_einfo_field_id_t k_tag;
        lccrt_einfo_field_id_t k_name;
        lccrt_einfo_field_id_t k_bitsize;
        lccrt_einfo_field_id_t k_bitoffset;
        lccrt_einfo_field_id_t k_align;
        //lccrt_einfo_field_id_t k_base;
        lccrt_einfo_field_id_t k_elems;
        //lccrt_einfo_field_id_t k_encoding;
        lccrt_einfo_field_id_t k_specific;
    };
    struct Subrange {
        lccrt_einfo_tydescr_ptr etyde;
        lccrt_einfo_field_id_t k_len;
        lccrt_einfo_field_id_t k_low;
        lccrt_einfo_field_id_t k_high;
        lccrt_einfo_field_id_t k_stride;
    };
    struct Gvar {
        lccrt_einfo_tydescr_ptr etyde;
        lccrt_einfo_field_id_t k_type;
        lccrt_einfo_field_id_t k_name;
        lccrt_einfo_field_id_t k_linkname;
    };

public:
    void reset( lccrt_module_ptr m);
    lccrt_einfo_category_t getDbg();
    ModuleData getModuleData();
    Type getType();
    Gvar getGvar();
    Subrange getSubrange();

private:
    lccrt_module_ptr m;
    lccrt_einfo_category_t edbg;
    struct {
        ModuleData module_data;
        Type type;
        Gvar gvar;
        Subrange subrange;
    } d;
};

/**
 * Генератор метаданных с отладочной информацией.
 */
class LLVM_LIBRARY_VISIBILITY LccrtDbgEmitter {
public:
    typedef std::map<const MDNode *, int64_t> MetaHash;
    typedef std::map<const MDNode *, lccrt_einfo_reference_t> EinfoHash;
    typedef std::map<StringRef, lccrt_einfo_reference_t> RawStringHash;

public:
    void makeModuleDbgMetadata( lccrt_module_ptr m, const Module *M);
    void makeGlobalVariableDbgMetadata( lccrt_var_ptr g, const GlobalVariable *G);
    void makeFunctionDbgMetadata( lccrt_function_ptr f, const Function *F);

private:
    void findChildsMetadata( const MDNode *);
    void findGlobalObjectMetadata( const GlobalObject *GO);
    void findModuleMetadata( const Module *M);

private:
    bool isMetadataI64( const Metadata *MD, int64_t &value);
    lccrt_einfo_reference_t makeI64( uint64_t value);
    lccrt_einfo_reference_t makeMetadataI64( const Metadata *MD);
    lccrt_einfo_reference_t makeRaw( const StringRef &value);
    lccrt_einfo_reference_t makeMetadataArrI64( const Metadata *MD);
    lccrt_einfo_reference_t makeDbgType( const DIType *T);
    lccrt_einfo_reference_t makeDbgGlobal( const DIGlobalVariable *G);
    lccrt_einfo_reference_t getModuleData();

private:
    lccrt_module_ptr m; /* модуль lccrt-IR */
    const Module *M; /* модуль llvm-IR */
    lccrt_function_ptr f; /* обрабатываемая функция */
    const Function *F; /* обрабатываемая функция */
    SmallVector<StringRef, 8> MDNames; /* название meta-данных по порядкову индексу */
    MetaHash metas; /* соответствие метаданные llvm-IR -> идентификатор */
    EinfoHash einfos; /* соответствие метаданные llvm-IR -> lccrt-IR */
    RawStringHash eraws; /* соответствие строка -> lccrt-IR */
    LccrtDbgEtydes etydes; /* множество описаний типов */
    lccrt_einfo_reference_t eref_mdata; /* данные модуля */
    lccrt_einfo_reference_t eref_types; /* массив всех dbg-типов */
    lccrt_einfo_reference_t eref_gvars; /* массив всех dbg-глобалов */
};

}

#endif /* LLVM_WITH_LCCRT */
#endif /* LLVM_LIB_CODEGEN_LCCRT_LCCRTDBG_H */
