/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file typesym.h
 ************************************************************************
 * Description:
 *  1byte char symbol vs type definition
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__TYPESYM_H
#define G__TYPESYM_H

/*    a b c d e f g h i j k l m n o p q r s t u v w x y z 0 1
 *    x X X X x X X X X   X X x   O O o X X x X x x o x x X     old
 *    x X X X x X X X X x X X     O O   X X x X x x o x x X x
 *                            X X     X
 */

#define G__BOOL_SYM       'g'

#define G__CHAR_SYM       'c'
#define G__UCHAR_SYM      'b'
#define G__SHORT_SYM      's'
#define G__USHORT_SYM     'r'
#define G__INT_SYM        'i'
#define G__UINT_SYM       'h'
#define G__LONG_SYM       'l'
#define G__ULONG_SYM      'k'

#define G__FLOAT_SYM      'f'
#define G__DOUBLE_SYM     'd'

#define G__LONGLONG_SYM   'n
#define G__ULONGLONG_SYM  'm'   <<<   
#define G__LONGDOUBLE_SYM 'q'   <<< 

#define G__LOGIC_SYM      'w'

#define G__CLASS_SYM      'u'

#define G__PVOID_SYM      'Y'
#define G__PFILE_SYM      'E'
#define G__P2F_SYM        'Q'   <<<  -> '1'
#define G__P2MEMF_SYM     'a'
#define G__SPECIALOBJ_SYM 'Z'
#define G__MACROSTR_SYM   'T'
#define G__MACRO_SYM      'm'   <<<  -> 'j'


#endif
