/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


/* from <windows.h> looks like */
typedef int DWORD;

typedef struct _tagSOME_STRUCT2 { int a; } SOME_STRUCT2, *PSOME_STRUCT2;
typedef struct _tagSOME_STRUCT3 { int b; int c; } SOME_STRUCT3, *PSOME_STRUCT3;

typedef struct _tagSOME_STRUCT
{
    DWORD some_member;

    union
   {
        /* the union members are structures which can contain additional structures.
	 * all structures have been forward declared (CINT/MAKECINT no errors)*/
        SOME_STRUCT2 s2;
        SOME_STRUCT3 s3;
    } u;

} SOME_STRUCT, *PSOME_STRUCT;

/* in <winfunc.h>: */

#ifdef __MAKECINT__

#pragma link off all functions;
#pragma link off all classes;
#pragma link off all globals;
#pragma link off all typedefs;


#pragma link C class _tagSOME_STRUCT;
#pragma link C typedef SOME_STRUCT;
#pragma link C typedef PSOME_STRUCT; 

#pragma link C nestedclasses;
#pragma link C nestedtypedefs;

#pragma link C class _tagSOME_STRUCT2;
#pragma link C typedef SOME_STRUCT2;
#pragma link C typedef PSOME_STRUCT2; 

#pragma link C class _tagSOME_STRUCT3;
#pragma link C typedef SOME_STRUCT3;
#pragma link C typedef PSOME_STRUCT3; 

#if G__CINTVERSION < 70000000
#pragma link C union SOME_STRUCT::; 
#else
#pragma link C union _tagSOME_STRUCT::*;
#endif


#endif

/* linked the nested structs in too */




