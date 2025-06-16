//===-- LccrtDbglinesEtydes.cpp - Common Lccrt code ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LccrtDbglinesEtydes class.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"

#ifdef LLVM_WITH_LCCRT
#include "LccrtDbglines.h"

using namespace llvm;

/**
 * Сброс текущего модуля.
 */
void
LccrtDbglinesEtydes::reset( lccrt_module_ptr vm) {
    m = vm;
    edbgl = lccrt_einfo_category_empty();
    memset( &d, 0, sizeof( d));

    return;
} /* LccrtDbglinesEtydes::reset */

/**
 * Получить dbg-категорию.
 */
lccrt_einfo_category_t
LccrtDbglinesEtydes::getDbg() {
    if ( !lccrt_einfo_category_is_valued( edbgl) ) {
        edbgl = lccrt_module_new_einfo_category( m, "dbg");
    }

    return (edbgl);
} /* LccrtDbglinesEtydes::getDbg */

/**
 * Получить описание типа для корня данных модуля.
 */
LccrtDbglinesEtydes::ModuleData
LccrtDbglinesEtydes::getModuleData() {
    if ( !d.module_data.etyde ) {
        std::vector<const char *> names;
        std::vector<lccrt_einfo_tydescr_ptr> types;
        lccrt_einfo_tydescr_ptr eraw = lccrt_einfo_make_tydescr_raw( m);
        lccrt_einfo_tydescr_ptr efns = lccrt_einfo_make_tydescr_array( m, getFunctionData().etyde);

        types.push_back( eraw); names.push_back( "file_name");
        types.push_back( eraw); names.push_back( "comp_dir");
        types.push_back( efns); names.push_back( "funcs");

        d.module_data.etyde = lccrt_einfo_make_tydescr_struct( m, "dbg_lines.module_data", names.size(),
                                                               names.data(), types.data());
        d.module_data.k_file_name = lccrt_einfo_find_tydescr_field( d.module_data.etyde, "file_name");
        d.module_data.k_comp_dir = lccrt_einfo_find_tydescr_field( d.module_data.etyde, "comp_dir");
        d.module_data.k_funcs = lccrt_einfo_find_tydescr_field( d.module_data.etyde, "funcs");
    }

    return (d.module_data);
} /* LccrtDbglinesEtydes::getModuleData */

LccrtDbglinesEtydes::FunctionData
LccrtDbglinesEtydes::getFunctionData() {
    if ( !d.function_data.etyde ) {
        std::vector<const char *> names;
        std::vector<lccrt_einfo_tydescr_ptr> types;
        lccrt_einfo_tydescr_ptr ei64_1 = lccrt_einfo_make_tydescr_i64( m);
        lccrt_einfo_tydescr_ptr eops = lccrt_einfo_make_tydescr_array( m, getOperData().etyde);

        types.push_back( ei64_1); names.push_back( "line");
        types.push_back( eops); names.push_back( "opers");

        d.function_data.etyde = lccrt_einfo_make_tydescr_struct( m, "dbg_lines.function_data", names.size(),
                                                                 names.data(), types.data());
        d.function_data.k_line = lccrt_einfo_find_tydescr_field( d.function_data.etyde, "line");
        d.function_data.k_opers = lccrt_einfo_find_tydescr_field( d.function_data.etyde, "opers");
    }

    return (d.function_data);
} /* LccrtDbgEtydes::getFunctionData */

LccrtDbglinesEtydes::OperData
LccrtDbglinesEtydes::getOperData() {
    if ( !d.oper_data.etyde ) {
        std::vector<const char *> names;
        std::vector<lccrt_einfo_tydescr_ptr> types;
        lccrt_einfo_tydescr_ptr ei64 = lccrt_einfo_make_tydescr_i64( m);

        types.push_back( ei64); names.push_back( "line");

        d.oper_data.etyde = lccrt_einfo_make_tydescr_struct( m, "dbg_lines.oper_data", names.size(),
                                                             names.data(), types.data());
        d.oper_data.k_line = lccrt_einfo_find_tydescr_field( d.oper_data.etyde, "line");
    }

    return (d.oper_data);
} /* LccrtDbglinesEtydes::getOperData */

#endif /* LLVM_WITH_LCCRT */
