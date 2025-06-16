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

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ModuleSlotTracker.h"

#ifdef LLVM_WITH_LCCRT
#include "LccrtDbg.h"

/* Структура дерава метаданных с отладочной информацией.
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
 * +--> DICompileUnit
 * +--> DISubprograms[struct] <= lccrt_function    // Массив с описанием функций
 *      |
 *      +--> name(str)                             // Имя функции
 *      +--> line(int)                             // Номер строки определения
 *      +--> types[struct]                         // Массив описаний типов возвращаемого
 *      |    |                                     // значения и аргументов функции
 *      |    +--> type_name(str)                   // Имя типа
 *      |    +--> type_size(int)                   // Размер значений типа (в битах)
 *      +--> DILocations[struct] <= lccrt_oper     // Массив с описанием операций функции
 *      |    |
 *      |    +--> line(int)                        // Номер строки операции
 *      |    +--> column(int)                      // Номер столбца операции
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
 * Добавить метаданные и их потомков в общее множество метеданных.
 */
void
LccrtDbgEmitter::findChildsMetadata( const MDNode *N) {
#if 0
    // Don't make slots for DIExpressions or DIArgLists. We just print them inline
    // everywhere.
    if (isa<DIExpression>(N) || isa<DIArgList>(N))
        return;
#endif

    if ( (metas.find( N) == metas.end()) ) {
        metas[N] = metas.size();

        // Recursively add any MDNodes referenced by operands.
        for ( unsigned i = 0, e = N->getNumOperands(); i != e; ++i ) {
            if ( const MDNode *Op = dyn_cast_or_null<MDNode>( N->getOperand( i)) ) {
                findChildsMetadata( Op);
            }
        }
    }

    return;
} /* LccrtDbgEmitter::findChildsMetadata */

/**
 * Помещение в указанный массив всех обнаруженных метаданных глобального объекта.
 */
void
LccrtDbgEmitter::findGlobalObjectMetadata( const GlobalObject *GO)
{
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;

    GO->getAllMetadata( MDs);
    for ( auto &MD : MDs ) {
        findChildsMetadata( MD.second);
    }

    return;
} /* LccrtDbgEmitter::findGlobalObjectMetadata */

/**
 * Помещение в указанный массив всех обнаруженных метаданных модуля.
 */
void
LccrtDbgEmitter::findModuleMetadata( const Module *M)
{
    M->getContext().getMDKindNames( MDNames);
    for ( const GlobalVariable &Var : M->globals() ) {
        findGlobalObjectMetadata( &Var);
    }

    // Add metadata used by named metadata.
    for ( const NamedMDNode &NMD : M->named_metadata() ) {
        for ( unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i ) {
            findChildsMetadata( NMD.getOperand( i));
        }
    }

    for ( const Function &F : *M )
    {
        const CallInst *CI = 0;
        MetadataAsValue *V = 0;
        MDNode *N = 0;

        findGlobalObjectMetadata( &F);
        for ( auto &BB : F ) {
            for ( auto &I : BB ) {
                // Для интрисика проверить аргументы.
                if ( (CI = dyn_cast<CallInst>( &I))
                     && CI->getCalledFunction()->isIntrinsic() ) {
                    for ( auto &Op : I.operands() ) {
                        if ( (V = dyn_cast_or_null<MetadataAsValue>( Op))
                             && (N = dyn_cast<MDNode>( V->getMetadata())) ) {
                            findChildsMetadata( N);
                        }
                    }
                }

                // Пройтись по всем метеданным связанным с операцией.
                SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
                I.getAllMetadata(MDs);
                for (auto &MD : MDs) {
                    findChildsMetadata( MD.second);
                }
            }
        }
    }

    return;
} /* LccrtDbgEmitter::findModuleMetadata */

/**
 * Если метаданных содержит целочисленную константу создать и вернуть i64-значение.
 */
bool
LccrtDbgEmitter::isMetadataI64( const Metadata *MD, int64_t &value) {
    const ConstantAsMetadata *VC = 0;
    const ConstantInt *VCI = 0;
    bool r = false;

    if ( MD
         && (VC = cast_or_null<ConstantAsMetadata>( MD))
         && (VCI = cast_or_null<ConstantInt>( VC->getValue())) ) {
        r = true;
        value = VCI->getSExtValue();
    }

    return (r);
} /* LccrtDbgEmitter::isMetadataI64 */

/**
 * Создать матаданные типа INT64.
 */
lccrt_einfo_reference_t
LccrtDbgEmitter::makeI64( uint64_t value) {
    lccrt_einfo_reference_t r = lccrt_einfo_new_i64( value);

    return (r);
} /* LccrtDbgEmitter::makeI64 */

/**
 * Создать матаданные типа RAW.
 */
lccrt_einfo_reference_t
LccrtDbgEmitter::makeRaw( const StringRef &value) {
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
} /* LccrtDbgEmitter::makeRaw */

/**
 * Если метаданных содержит целочисленную константу создать и вернуть i64-значение.
 */
lccrt_einfo_reference_t
LccrtDbgEmitter::makeMetadataI64( const Metadata *MD) {
    int64_t value;
    lccrt_einfo_reference_t r = lccrt_einfo_new_empty();

    if ( isMetadataI64( MD, value) ) {
        r = makeI64( value);
    }

    return (r);
} /* LccrtDbgEmitter::makeMetadataI64 */

/**
 * Создание массива идентификаторов метаданных.
 */
lccrt_einfo_reference_t
LccrtDbgEmitter::makeMetadataArrI64( const Metadata *MD) {
    lccrt_einfo_reference_t r = lccrt_einfo_new_array( etydes.getType().etyde_ai64, 0);
    auto NE = cast_or_null<MDTuple>( MD);

    for ( unsigned mi = 0, me = NE->getNumOperands(); mi != me; ++mi ) {
        const DIType *T = dyn_cast_or_null<DIType>( NE->getOperand( mi));

        if ( T ) {
            makeDbgType( T);
            lccrt_einfo_push_elem( r, makeI64( metas[T]));
        } else {
            lccrt_einfo_push_elem( r, lccrt_einfo_new_empty());
        }
    }

    return (r);
} /* LccrtDbgEmitter::makeMetadataArrI64 */

/**
 * Создать данные для модуля.
 */
lccrt_einfo_reference_t
LccrtDbgEmitter::getModuleData() {
    if ( lccrt_einfo_is_empty( eref_mdata) ) {
        LccrtDbgEtydes::ModuleData mdata = etydes.getModuleData();
        LccrtDbgEtydes::Type type = etydes.getType();
        LccrtDbgEtydes::Gvar gvar = etydes.getGvar();

        eref_mdata = lccrt_einfo_new_struct( mdata.etyde);
        eref_types = lccrt_einfo_new_array( lccrt_einfo_make_tydescr_array( m, type.etyde), 0);
        eref_gvars = lccrt_einfo_new_array( lccrt_einfo_make_tydescr_array( m, gvar.etyde), 0);

        lccrt_module_set_einfo( m, etydes.getDbg(), eref_mdata);
        lccrt_einfo_set_field( eref_mdata, mdata.k_types, eref_types);
        lccrt_einfo_set_field( eref_mdata, mdata.k_gvars, eref_gvars);
    }

    return (eref_mdata);
} /* LccrtDbgEmitter::getModuleData */

/**
 * Создать метаданные для DIType.
 */
lccrt_einfo_reference_t
LccrtDbgEmitter::makeDbgType( const DIType *T) {
    auto it = einfos.find( T);
    lccrt_einfo_reference_t r = lccrt_einfo_new_empty();
   
    if ( (it != einfos.end()) ) {
        r = it->second;
    } else {
        LccrtDbgEtydes::Type tb = etydes.getType();

        r = lccrt_einfo_new_struct( tb.etyde);
        einfos[T] = r;
        lccrt_einfo_set_field( r, tb.k_ident, makeI64( metas[T]));
        lccrt_einfo_set_field( r, tb.k_tag, makeRaw( dwarf::TagString( T->getTag())));
        lccrt_einfo_set_field( r, tb.k_name, makeRaw( T->getName()));
        lccrt_einfo_set_field( r, tb.k_bitsize, makeI64( T->getSizeInBits()));
        lccrt_einfo_set_field( r, tb.k_bitoffset, makeI64( T->getOffsetInBits()));
        lccrt_einfo_set_field( r, tb.k_align, makeI64( T->getAlignInBytes()));
        //lccrt_einfo_set_field( r, tb.k_base, makeI64( 0));
        //lccrt_einfo_set_field( r, tb.k_elems, makeI64( 0));

        if ( const auto *TB = dyn_cast<DIBasicType>( T) ) {
            StringRef encoding = dwarf::AttributeEncodingString( TB->getEncoding());

            lccrt_einfo_set_field( r, tb.k_type, makeRaw( "Basic"));
            //lccrt_einfo_set_field( r, tb.k_encoding, makeRaw( encoding));
            lccrt_einfo_set_field( r, tb.k_specific, makeRaw( encoding));

        } else if ( const auto *TD = dyn_cast<DIDerivedType>( T) ) {
            if ( (T->getTag() == dwarf::DW_TAG_pointer_type) ) {
                lccrt_einfo_set_field( r, tb.k_type, makeRaw( "Pointer"));
            } else if ( (T->getTag() == dwarf::DW_TAG_member) ) {
                lccrt_einfo_set_field( r, tb.k_type, makeRaw( "Field"));
            } else {
                assert( 0);
            }

            if ( DIType *TDB = TD->getBaseType() ) {
                makeDbgType( TDB);
                lccrt_einfo_set_field( r, tb.k_elems, makeI64( metas[TDB]));
            }
        } else if ( const auto *TC = dyn_cast<DICompositeType>( T) ) {
            MDTuple *TCE = cast_or_null<MDTuple>( TC->getRawElements());

            if ( (T->getTag() == dwarf::DW_TAG_array_type) ) {
                assert( TC->getRawElements() == 0);
                if ( (TCE->getNumOperands() == 1) ) {
                    DISubrange *S = dyn_cast<DISubrange>( TCE->getOperand( 0));
                    lccrt_einfo_reference_t ci64 = makeMetadataI64( S ? S->getRawCountNode() : 0);

                    if ( lccrt_einfo_is_valued( ci64)
                         && S->getRawCountNode()
                         && !S->getRawLowerBound()
                         && !S->getRawUpperBound()
                         && !S->getRawStride() ) {
                        //LccrtDbgEtydes::Subrange ts = etydes.getSubrange();

                        //lccrt_einfo_set_field( s, ts.k_len, S->getRawLowerBound());

                        lccrt_einfo_set_field( r, tb.k_type, makeRaw( "Array"));
                        lccrt_einfo_set_field( r, tb.k_specific, ci64);
                        //lccrt_einfo_set_field( r, tb.k_specific, s);
                    }
                }
            } else if ( (T->getTag() == dwarf::DW_TAG_structure_type) ) {
                assert( TC->getBaseType() == 0);
                lccrt_einfo_set_field( r, tb.k_type, makeRaw( "Struct"));
                lccrt_einfo_set_field( r, tb.k_elems, makeMetadataArrI64( TC->getRawElements()));
            } else {
                //printf( "DICompositeType %p\n", (const void *)T);
                //printf( "  elements: %p\n", (const void *)TC->getRawElements());
                assert( 0);
            }

            if ( DIType *TCB = TC->getBaseType() ) {
                makeDbgType( TCB);
                lccrt_einfo_set_field( r, tb.k_elems, makeI64( metas[TCB]));
            }
        } else if ( const auto *TS = dyn_cast<DISubroutineType>( T) ) {
            lccrt_einfo_set_field( r, tb.k_type, makeRaw( "Function"));
            lccrt_einfo_set_field( r, tb.k_elems, makeMetadataArrI64( TS->getRawTypeArray()));
        } else {
            printf( "DIType %p\n", (const void *)T);
            assert( 0);
        }

        getModuleData();
        lccrt_einfo_push_elem( eref_types, r);
    }

    return (r);
} /* LccrtDbgEmitter::makeDbgType */

/**
 * Создать метаданные для DIGlobalVariable.
 */
lccrt_einfo_reference_t
LccrtDbgEmitter::makeDbgGlobal( const DIGlobalVariable *G) {
    auto it = einfos.find( G);
    lccrt_einfo_reference_t r = lccrt_einfo_new_empty();
   
    printf( "DIGV: %p\n", G);

    if ( (it != einfos.end()) ) {
        r = it->second;
    } else {
        LccrtDbgEtydes::Gvar tg = etydes.getGvar();

        r = lccrt_einfo_new_struct( tg.etyde);
        einfos[G] = r;
        lccrt_einfo_set_field( r, tg.k_type, makeDbgType( G->getType()));
        lccrt_einfo_set_field( r, tg.k_name, makeRaw( G->getName()));
        lccrt_einfo_set_field( r, tg.k_linkname, makeRaw( G->getLinkageName()));

        getModuleData();
        lccrt_einfo_push_elem( eref_gvars, r);
    }

    return (r);
} /* LccrtDbgEmitter::makeDbgGlobal */

/**
 * Добавление отладочной информации о модуле.
 */
void
LccrtDbgEmitter::makeModuleDbgMetadata( lccrt_module_ptr vm, const Module *vM)
{
    m = vm;
    M = vM;
    metas = {};
    einfos = {};
    eraws = {};
    etydes.reset( m);
    eref_mdata = lccrt_einfo_new_empty();
    eref_types = lccrt_einfo_new_empty();
    eref_gvars = lccrt_einfo_new_empty();

    return;

    findModuleMetadata( M);

    for ( auto &it : metas ) {
        const MDNode *N = it.first;

        if ( const auto *T = dyn_cast<DIType>( N) ) {
            makeDbgType( T);
        } else if ( const auto *G = dyn_cast<DIGlobalVariable>( N) ) {
            makeDbgGlobal( G);
        }
    }

    for ( const NamedMDNode &Node : M->named_metadata() ) {
        //printNamedMDNode( &Node);
        fprintf( stderr, "%s\n", Node.getName().data());
    }

    // Upgrade list of variables attached to the CUs.
    if ( NamedMDNode *CUNodes = M->getNamedMetadata( "llvm.dbg.cu") ) {
        for ( unsigned I = 0, E = CUNodes->getNumOperands(); I != E; ++I ) {
            auto *CU = cast<DICompileUnit>( CUNodes->getOperand( I));
            unsigned src_lang = CU->getSourceLanguage();

            printf( "DICompileUnit %p\n", (const void *)CU);
            if ( src_lang ) printf( "  language: %s\n", dwarf::LanguageString( src_lang).data());
            printf( "  file: %p\n", (const void *)CU->getRawFile());
            printf( "  enums: %p\n", (const void *)CU->getRawEnumTypes());
            printf( "  globals: %p\n", (const void *)CU->getRawGlobalVariables());
            //Printer.printMetadata( "enums", N->getRawEnumTypes());

            if ( CU->getRawFile() )
            {
                auto *F = cast<DIFile>( CU->getRawFile());

                printf( "DIFile %p\n", (const void *)F);
                printf( "  filename: %s\n", F->getFilename().data());
                printf( "  directory: %s\n", F->getDirectory().data());
            }

            if ( CU->getRawGlobalVariables() )
            {
                const MDTuple *GVS = cast<MDTuple>( CU->getRawGlobalVariables());

                printf( "GlobalVariables %p\n", (const void *)GVS);
                for ( unsigned mi = 0, me = GVS->getNumOperands(); mi != me; ++mi) {
                    const Metadata *GVSi = GVS->getOperand( mi);

                    printf( "  %p\n", (void *)GVSi);
                }

                for ( unsigned mi = 0, me = GVS->getNumOperands(); mi != me; ++mi) {
                    auto *GVE = cast<DIGlobalVariableExpression>( GVS->getOperand( mi));

                    if ( GVE )
                    {
                        auto *GV = cast<DIGlobalVariable>( GVE->getRawVariable());

                        printf( "DIGlobalVariableExpression %p\n", (void *)GVE);
                        printf( "  variable: %p\n", (void *)GVE->getRawVariable());
                        if ( GV )
                        {
                            printf( "DIGlobalVariable %p\n", (void *)GV);
                            printf( "  name: %s\n", GV->getName().data());
                            printf( "  file: %p\n", (const void *)GV->getRawFile());
                            printf( "  line: %d\n", GV->getLine());
                            printf( "  type: %p\n", (const void *)GV->getRawType());
                        }
                    }
                }
            }
        }
    }

    return;
} /* LccrtDbgEmitter::makeModuleDbgMetadata */

/**
 * Добавление отладочной информации о переменной.
 */
void
LccrtDbgEmitter::makeGlobalVariableDbgMetadata( lccrt_var_ptr g, const GlobalVariable *G)
{
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    G->getAllMetadata( MDs);

    for ( const auto &I : MDs ) {
        auto *GVE = dyn_cast<DIGlobalVariableExpression>( I.second);

        if ( GVE ) {
            auto it = einfos.find( GVE->getVariable());

            if ( (it != einfos.end()) ) {
                assert( !lccrt_einfo_is_valued( lccrt_var_get_einfo( g, etydes.getDbg())));
                lccrt_var_set_einfo( g, etydes.getDbg(), it->second);
            }
        }
    }

    return;
} /* LccrtDbgEmitter::makeGlobalVariableDbgMetadata */

/**
 * Добавление отладочной информации о модуле.
 */
void
LccrtDbgEmitter::makeFunctionDbgMetadata( lccrt_function_ptr vf, const Function *vF)
{
    f = vf;
    F = vF;

    //SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    //F->getAllMetadata( MDs);
    //printMetadataAttachments(MDs, " ");

    return;
} /* LccrtDbgEmitter::makeFunctionDbgMetadata */

#endif /* LLVM_WITH_LCCRT */
