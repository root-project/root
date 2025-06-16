//===-- LccrtDbg.cpp - Common Lccrt code ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LccrtDbg class.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"

#ifdef LLVM_WITH_LCCRT
#include "LccrtDbg.h"

using namespace llvm;

/**
 * Сброс текущего модуля.
 */
void
LccrtDbgEtydes::reset( lccrt_module_ptr vm) {
    m = vm;
    edbg = lccrt_einfo_category_empty();
    memset( &d, 0, sizeof( d));

    return;
} /* LccrtDbgEtydes::reset */

/**
 * Получить dbg-категорию.
 */
lccrt_einfo_category_t
LccrtDbgEtydes::getDbg() {
    if ( !lccrt_einfo_category_is_valued( edbg) ) {
        edbg = lccrt_module_new_einfo_category( m, "dbg");
    }

    return (edbg);
} /* LccrtDbgEtydes::getDbg */

/**
 * Получить описание типа для корня данных модуля.
 */
LccrtDbgEtydes::ModuleData
LccrtDbgEtydes::getModuleData() {
    if ( !d.module_data.etyde ) {
        std::vector<const char *> names;
        std::vector<lccrt_einfo_tydescr_ptr> types;
        lccrt_einfo_tydescr_ptr ei64 = lccrt_einfo_make_tydescr_i64( m);
        lccrt_einfo_tydescr_ptr eraw = lccrt_einfo_make_tydescr_raw( m);
        lccrt_einfo_tydescr_ptr etps = lccrt_einfo_make_tydescr_array( m, getType().etyde);
        lccrt_einfo_tydescr_ptr egvs = lccrt_einfo_make_tydescr_array( m, getGvar().etyde);

        types.push_back( etps); names.push_back( "types");
        types.push_back( egvs); names.push_back( "gvars");

        d.module_data.etyde = lccrt_einfo_make_tydescr_struct( m, "dbg.module_data", names.size(),
                                                               names.data(), types.data());
        d.module_data.k_types = lccrt_einfo_find_tydescr_field( d.module_data.etyde, "types");
        d.module_data.k_gvars = lccrt_einfo_find_tydescr_field( d.module_data.etyde, "gvars");
    }

    return (d.module_data);
} /* LccrtDbgEtydes::getModuleData */

/**
 * Получить описание типа для DIType.
 */
LccrtDbgEtydes::Type
LccrtDbgEtydes::getType() {
    if ( !d.type.etyde ) {
        std::vector<const char *> names;
        std::vector<lccrt_einfo_tydescr_ptr> types;
        lccrt_einfo_tydescr_ptr ei64 = lccrt_einfo_make_tydescr_i64( m);
        lccrt_einfo_tydescr_ptr eraw = lccrt_einfo_make_tydescr_raw( m);
        lccrt_einfo_tydescr_ptr earr = lccrt_einfo_make_tydescr_array( m, ei64);
        lccrt_einfo_tydescr_ptr eelm_types[2] = {ei64,earr};
        lccrt_einfo_tydescr_ptr eelm = lccrt_einfo_make_tydescr_union( m, 2, eelm_types);
        lccrt_einfo_tydescr_ptr esre = getSubrange().etyde;
        lccrt_einfo_tydescr_ptr espf_types[3] = {eraw,ei64,esre};
        lccrt_einfo_tydescr_ptr espf = lccrt_einfo_make_tydescr_union( m, 3, espf_types);

        types.push_back( ei64); names.push_back( "ident");
        types.push_back( eraw); names.push_back( "type");
        types.push_back( eraw); names.push_back( "tag");
        types.push_back( eraw); names.push_back( "name");
        types.push_back( ei64); names.push_back( "bitsize");
        types.push_back( ei64); names.push_back( "bitoffset");
        types.push_back( ei64); names.push_back( "align");
        //types.push_back( ei64); names.push_back( "base");
        types.push_back( eelm); names.push_back( "elems");
        //types.push_back( eraw); names.push_back( "encoding");
        types.push_back( espf); names.push_back( "specific");

        d.type.etyde_ai64 = earr;
        d.type.etyde = lccrt_einfo_make_tydescr_struct( m, "dbg.type", names.size(),
                                                        names.data(), types.data());
        d.type.k_ident     = lccrt_einfo_find_tydescr_field( d.type.etyde, "ident");
        d.type.k_type      = lccrt_einfo_find_tydescr_field( d.type.etyde, "type");
        d.type.k_tag       = lccrt_einfo_find_tydescr_field( d.type.etyde, "tag");
        d.type.k_name      = lccrt_einfo_find_tydescr_field( d.type.etyde, "name");
        d.type.k_bitsize   = lccrt_einfo_find_tydescr_field( d.type.etyde, "bitsize");
        d.type.k_bitoffset = lccrt_einfo_find_tydescr_field( d.type.etyde, "bitoffset");
        d.type.k_align     = lccrt_einfo_find_tydescr_field( d.type.etyde, "align");
        //d.type.k_base      = lccrt_einfo_find_tydescr_field( d.type.etyde, "base");
        d.type.k_elems     = lccrt_einfo_find_tydescr_field( d.type.etyde, "elems");
        //d.type.k_encoding  = lccrt_einfo_find_tydescr_field( d.type.etyde, "encoding");
        d.type.k_specific  = lccrt_einfo_find_tydescr_field( d.type.etyde, "specific");
    }

    return (d.type);
} /* LccrtDbgEtydes::getType */

/**
 * Получить описание типа для DISubrange.
 */
LccrtDbgEtydes::Subrange
LccrtDbgEtydes::getSubrange() {
    if ( !d.subrange.etyde ) {
        std::vector<const char *> names;
        std::vector<lccrt_einfo_tydescr_ptr> types;
        lccrt_einfo_tydescr_ptr ei64 = lccrt_einfo_make_tydescr_i64( m);

        types.push_back( ei64); names.push_back( "len");
        types.push_back( ei64); names.push_back( "low");
        types.push_back( ei64); names.push_back( "high");
        types.push_back( ei64); names.push_back( "stride");

        d.subrange.etyde = lccrt_einfo_make_tydescr_struct( m, "dbg.subrange", names.size(),
                                                            names.data(), types.data());
        d.subrange.k_len    = lccrt_einfo_find_tydescr_field( d.subrange.etyde, "len");
        d.subrange.k_low    = lccrt_einfo_find_tydescr_field( d.subrange.etyde, "low");
        d.subrange.k_high   = lccrt_einfo_find_tydescr_field( d.subrange.etyde, "high");
        d.subrange.k_stride = lccrt_einfo_find_tydescr_field( d.subrange.etyde, "stride");
    }

    return (d.subrange);
} /* LccrtDbgEtydes::getSubrange */

/**
 * Получить описание типа для глобальной переменной.
 */
LccrtDbgEtydes::Gvar
LccrtDbgEtydes::getGvar() {
    if ( !d.gvar.etyde ) {
        std::vector<const char *> names;
        std::vector<lccrt_einfo_tydescr_ptr> types;
        lccrt_einfo_tydescr_ptr ei64 = lccrt_einfo_make_tydescr_i64( m);
        lccrt_einfo_tydescr_ptr eraw = lccrt_einfo_make_tydescr_raw( m);
        lccrt_einfo_tydescr_ptr etdr = getType().etyde;

        types.push_back( etdr); names.push_back( "type");
        types.push_back( eraw); names.push_back( "name");
        types.push_back( eraw); names.push_back( "linkname");

        d.gvar.etyde = lccrt_einfo_make_tydescr_struct( m, "dbg.gvar", names.size(),
                                                        names.data(), types.data());
        d.gvar.k_type = lccrt_einfo_find_tydescr_field( d.gvar.etyde, "type");
        d.gvar.k_name = lccrt_einfo_find_tydescr_field( d.gvar.etyde, "name");
        d.gvar.k_linkname = lccrt_einfo_find_tydescr_field( d.gvar.etyde, "linkname");
    }

    return (d.gvar);
} /* LccrtDbgEtydes::getGvar */

#endif /* LLVM_WITH_LCCRT */
