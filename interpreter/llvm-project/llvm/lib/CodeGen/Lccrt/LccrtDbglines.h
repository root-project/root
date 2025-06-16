//=- llvm/CodeGen/Lccrt/LccrtDbglines.h - Lccrt-IR translation -*- C++ -*-=//
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

#ifndef LLVM_LIB_CODEGEN_LCCRT_LCCRTDBGLINES_H
#define LLVM_LIB_CODEGEN_LCCRT_LCCRTDBGLINES_H

#include <map>

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"

#ifdef LLVM_WITH_LCCRT
#include "lccrt.h"

namespace llvm {

class LccrtDbglinesEtydes {
public:
    struct ModuleData {
        lccrt_einfo_tydescr_ptr etyde;
        lccrt_einfo_field_id_t k_file_name;
        lccrt_einfo_field_id_t k_comp_dir;
        lccrt_einfo_field_id_t k_funcs;
    };
    struct FunctionData {
        lccrt_einfo_tydescr_ptr etyde;
        lccrt_einfo_field_id_t k_file;
        lccrt_einfo_field_id_t k_line;
        lccrt_einfo_field_id_t k_opers;
    };
    struct OperData {
        lccrt_einfo_tydescr_ptr etyde;
        lccrt_einfo_field_id_t k_line;
    };

public:
    void reset( lccrt_module_ptr m);
    lccrt_einfo_category_t getDbg();
    ModuleData getModuleData();
    FunctionData getFunctionData();
    OperData getOperData();

private:
    lccrt_module_ptr m;
    lccrt_einfo_category_t edbgl;
    struct {
        ModuleData module_data;
        FunctionData function_data;
        OperData oper_data;
    } d;
};

/**
 * ��������� ���������� � ���������� �����������.
 */
class LLVM_LIBRARY_VISIBILITY LccrtDbglinesEmitter {
public:
    typedef std::map<const MDNode *, int64_t> MetaHash;
    typedef std::map<const MDNode *, lccrt_einfo_reference_t> EinfoHash;
    typedef std::map<StringRef, lccrt_einfo_reference_t> RawStringHash;

public:
    void makeModuleDbgMetadata( lccrt_module_ptr m, const Module *M);
    void makeFunctionDbgMetadata( lccrt_function_ptr f, const Function *F);
    void makeOperDbgMetadata( lccrt_oper_ptr o, const Instruction *I);

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
    lccrt_einfo_reference_t getModuleData();
    lccrt_einfo_reference_t getFunctionData();
    lccrt_einfo_reference_t getOperData();

private:
    lccrt_module_ptr m; /* ������ lccrt-IR */
    const Module *M; /* ������ llvm-IR */
    lccrt_function_ptr f; /* �������������� ������� */
    const Function *F; /* �������������� ������� */
    lccrt_oper_ptr o; /* �������������� �������� */
    const Instruction *I; /* �������������� �������� */
    SmallVector<StringRef, 8> MDNames; /* �������� meta-������ �� ��������� ������� */
    MetaHash metas; /* ������������ ���������� llvm-IR -> ������������� */
    EinfoHash einfos; /* ������������ ���������� llvm-IR -> lccrt-IR */
    RawStringHash eraws; /* ������������ ������ -> lccrt-IR */
    LccrtDbglinesEtydes etydes; /* ��������� �������� ����� */
    lccrt_einfo_reference_t eref_mdata; /* ������ ������ */
    lccrt_einfo_reference_t eref_fdata; /* ������ ������� */
    lccrt_einfo_reference_t eref_odata; /* ������ �������� */
    lccrt_einfo_reference_t eref_funcs; /* ������ ���� ������� */
    lccrt_einfo_reference_t eref_opers; /* ������ ���� �������� */
};

}

#endif /* LLVM_WITH_LCCRT */
#endif /* LLVM_LIB_CODEGEN_LCCRT_LCCRTDBGLINES_H */
