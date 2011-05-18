/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Property.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__PROPERTY_H
#define G__PROPERTY_H

/* Normal Property() */

#define G__BIT_ISTAGNUM          0x0000000f
#define G__BIT_ISCLASS           0x00000001
#define G__BIT_ISSTRUCT          0x00000002
#define G__BIT_ISUNION           0x00000004
#define G__BIT_ISENUM            0x00000008
#define G__BIT_ISTYPEDEF         0x00000010
#define G__BIT_ISFUNDAMENTAL     0x00000020
#define G__BIT_ISABSTRACT        0x00000040
#define G__BIT_ISVIRTUAL         0x00000080
#define G__BIT_ISPUREVIRTUAL     0x00000100
#define G__BIT_ISPUBLIC          0x00000200
#define G__BIT_ISPROTECTED       0x00000400
#define G__BIT_ISPRIVATE         0x00000800
#define G__BIT_ISPOINTER         0x00001000
#define G__BIT_ISARRAY           0x00002000
#define G__BIT_ISSTATIC          0x00004000
#define G__BIT_ISDEFAULT         0x00008000

#define G__BIT_ISREFERENCE       0x00010000
#define G__BIT_ISDIRECTINHERIT   0x00020000
#define G__BIT_ISCCOMPILED       0x00040000
#define G__BIT_ISCPPCOMPILED     0x00080000
#define G__BIT_ISCOMPILED        0x000c0000
#define G__BIT_ISBYTECODE        0x02000000
#define G__BIT_ISCONSTANT        0x00100000
#define G__BIT_ISVIRTUALBASE     0x00200000
#define G__BIT_ISPCONSTANT       0x00400000
#define G__BIT_ISMETHCONSTANT    0x10000000 // method is const

#define G__BIT_ISGLOBALVAR       0x00800000
#define G__BIT_ISLOCALVAR        0x01000000
#define G__BIT_ISEXPLICIT        0x04000000
#define G__BIT_ISNAMESPACE       0x08000000

#define G__BIT_ISUSINGVARIABLE   0x20000000



/* ECF enhancement  ClassProperty() */

#define G__CLS_VALID             0x00000001

#define G__CLS_HASEXPLICITCTOR   0x00000010
#define G__CLS_HASIMPLICITCTOR   0x00000020
#define G__CLS_HASCTOR           0x00000030
#define G__CLS_HASDEFAULTCTOR    0x00000040
#define G__CLS_HASASSIGNOPR      0x00000080

#define G__CLS_HASEXPLICITDTOR   0x00000100
#define G__CLS_HASIMPLICITDTOR   0x00000200
#define G__CLS_HASDTOR           0x00000300

#define G__CLS_HASVIRTUAL        0x00001000
#define G__CLS_ISABSTRACT        0x00002000


#endif /* G__PROPERTY_H */
