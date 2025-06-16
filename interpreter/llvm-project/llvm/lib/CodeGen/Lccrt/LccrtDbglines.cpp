//===-- LccrtDbglines.cpp - Common Lccrt code ---------------------------===//
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

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ModuleSlotTracker.h"

#ifdef LLVM_WITH_LCCRT
#include "LccrtDbglines.h"

/* Структура дерева метаданных с отладочной информацией.
 *
 * ! Обновлять этот комментарий вместе с изменениями в этом файле !
 *
 * Условные обозначения:
 * N <= IR -- IR (lccrt_module, lccrt_function или lccrt_oper)
 *            ссылается на узел N метаданных
 * N[T]    -- узел N является массивом, элементы которго имеют тип T
 * N(str)  -- узел N является строкой
 * N(int)  -- узел N является числом
 *
 * llvm_dbg <= lccrt_module
 * |
 * +--> DICompileUnit <= lccrt_module              // Модуль
 *      |
 *      +--> file_name(str)                        // Имя исходного файла
 *      +--> comp_dir(str)                         // Директория компиляции
 * +--> DISubprograms[struct] <= lccrt_function    // Массив с описанием функций
 *      |
 *      +--> name(str)                             // Имя функции
 *      +--> line(int)                             // Номер строки лексического блока
 *      +--> types[struct]                         // Массив описаний типов возвращаемого
 *      |    |                                     // значения и аргументов функции
 *      |    +--> type_name(str)                   // Имя типа
 *      |    +--> type_size(int)                   // Размер значений типа (в битах)
 *      +--> DILocations[struct] <= lccrt_oper     // Массив с описанием операций функции
 *      |    |
 *      |    +--> line(int)                        // Номер строки операции
 *      +--> DILovalVariables[struct]              // Массив с описанием локальных
 *           |                                     // переменных функции
 *           |
 *           +--> name(str)                        // Имя переменной
 *           +--> line(int)                        // Номер строки объявления переменной
 *           +--> type_name(str)                   // Имя типа
 *           +--> type_size(int)                   // Размер значений типа (в битах)
 */

using namespace llvm;

/**
 * Создать матаданные типа INT64.
 */
lccrt_einfo_reference_t
LccrtDbglinesEmitter::makeI64( uint64_t value) {
    lccrt_einfo_reference_t r = lccrt_einfo_new_i64( value);

    return (r);
} /* LccrtDbglinesEmitter::makeI64 */

/**
 * Создать матаданные типа RAW.
 */
lccrt_einfo_reference_t
LccrtDbglinesEmitter::makeRaw( const StringRef &value) {
    lccrt_einfo_reference_t r;
   
    if ( value.empty() ) {
        r = lccrt_einfo_new_empty();
    } else {
        auto it = eraws.find( value);

        if ( (it != eraws.end()) ) {
            r = it->second;
        } else {
            r = lccrt_einfo_new_raw_by_string( m, value.data());
            eraws[value] = r;
        }
    }

    return (r);
} /* LccrtDbglinesEmitter::makeRaw */

/**
 * Создать данные для модуля.
 */
lccrt_einfo_reference_t
LccrtDbglinesEmitter::getModuleData() {
    std::string file_name, comp_dir_name;
    std::string::size_type pos;

    if ( lccrt_einfo_is_empty( eref_mdata) ) {
        LccrtDbglinesEtydes::ModuleData mdata = etydes.getModuleData();
        LccrtDbglinesEtydes::FunctionData fdata = etydes.getFunctionData();

        eref_mdata = lccrt_einfo_new_struct( mdata.etyde);
        eref_funcs = lccrt_einfo_new_array( lccrt_einfo_make_tydescr_array( m, fdata.etyde), 0);

        file_name = M->getSourceFileName();

        pos = file_name.rfind( "/");
        if ( pos != std::string::npos )
        {
            comp_dir_name = file_name.substr(0, pos);
            file_name = file_name.substr(pos + 1);
        } else
        {
            comp_dir_name = ".";
        }

        lccrt_module_set_einfo( m, etydes.getDbg(), eref_mdata);
        lccrt_einfo_set_field( eref_mdata, mdata.k_file_name, makeRaw( file_name.c_str()));
        lccrt_einfo_set_field( eref_mdata, mdata.k_comp_dir, makeRaw( comp_dir_name.c_str()));
    }

    return (eref_mdata);
} /* LccrtDbglinesEmitter::getModuleData */

lccrt_einfo_reference_t
LccrtDbglinesEmitter::getFunctionData() {
    if ( lccrt_einfo_is_empty( eref_fdata) ) {
        LccrtDbglinesEtydes::FunctionData fdata = etydes.getFunctionData();
        LccrtDbglinesEtydes::OperData odata = etydes.getOperData();

        eref_fdata = lccrt_einfo_new_struct( fdata.etyde);
        eref_opers = lccrt_einfo_new_array( lccrt_einfo_make_tydescr_array( m, odata.etyde), 0);

        lccrt_function_set_einfo( f, etydes.getDbg(), eref_fdata);
        lccrt_einfo_set_field( eref_fdata, fdata.k_opers, eref_opers);
    }

    return (eref_fdata);
} /* LccrtDbglinesEmitter::getFunctionData */

lccrt_einfo_reference_t
LccrtDbglinesEmitter::getOperData() {
    if ( lccrt_einfo_is_empty( eref_odata) ) {
        LccrtDbglinesEtydes::OperData odata = etydes.getOperData();

        eref_odata = lccrt_einfo_new_struct( odata.etyde);

        lccrt_oper_set_einfo( o, etydes.getDbg(), eref_odata);
    }

    return (eref_odata);
} /* LccrtDbglinesEmitter::getOperData */

/**
 * Добавление отладочной информации о модуле.
 */
void
LccrtDbglinesEmitter::makeModuleDbgMetadata( lccrt_module_ptr vm, const Module *vM)
{
    m = vm;
    M = vM;
    metas = {};
    einfos = {};
    eraws = {};
    etydes.reset( m);
    eref_mdata = lccrt_einfo_new_empty();
    eref_fdata = lccrt_einfo_new_empty();
    eref_odata = lccrt_einfo_new_empty();

    getModuleData();

    //LccrtDbglinesEtydes::ModuleData mdata = etydes.getModuleData();

    return;
} /* LccrtDbglinesEmitter::makeModuleDbgMetadata */

/**
 * Добавление отладочной информации о функции.
 */
void
LccrtDbglinesEmitter::makeFunctionDbgMetadata( lccrt_function_ptr vf, const Function *vF)
{
    const DISubprogram *subprogram = vF->getSubprogram();

    if (subprogram) {
        f = vf;
        F = vF;

        getFunctionData();

        SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
        F->getAllMetadata( MDs);

        LccrtDbglinesEtydes::FunctionData fdata = etydes.getFunctionData();

        lccrt_einfo_set_field( eref_fdata, fdata.k_line, makeI64( subprogram->getScopeLine()));

        lccrt_einfo_push_elem( eref_funcs, eref_fdata);

        eref_fdata = lccrt_einfo_new_empty();
    }
} /* LccrtDbglinesEmitter::makeFunctionDbgMetadata */

void
LccrtDbglinesEmitter::makeOperDbgMetadata( lccrt_oper_ptr vo, const Instruction *vI)
{
    o = vo;
    I = vI;

    const DebugLoc &loc = I->getDebugLoc();
    const DILocation *node = dyn_cast_or_null<DILocation>( loc.getAsMDNode());

    if ( node ) {
        LccrtDbglinesEtydes::OperData odata = etydes.getOperData();
        lccrt_einfo_reference_t r;
        auto it = einfos.find( node);

        if ( (it != einfos.end()) ) {
            r = it->second;
        } else {
            r = getOperData();

            einfos[node] = r;
            lccrt_einfo_set_field( r, odata.k_line, makeI64( loc.getLine()));
        }

        lccrt_einfo_push_elem( eref_opers, r);

        eref_odata = lccrt_einfo_new_empty();
    }
} /* LccrtDbglinesEmitter::makeOperDbgMetadata */

#endif /* LLVM_WITH_LCCRT */
