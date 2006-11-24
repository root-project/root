/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file memtest.h
 ************************************************************************
 * Description:
 *  Memory leak test utility header file.
 * Comment:
 *  This header is usually un-included.
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__MEMTEST_H
#define G__MEMTEST_H

/* Turns on memory leak test */
#define G__MEMTEST

/*************************************************************************
* macro replacement
*************************************************************************/
#undef malloc
#undef calloc
#undef free
#undef realloc
#undef fopen
#undef fclose
#undef tmpfile

void *G__TEST_Malloc G__P((size_t size));
void *G__TEST_Calloc G__P((size_t n,size_t bsize));
void G__TEST_Free G__P((void* p));
void *G__TEST_Realloc G__P((void* p,size_t size));
int G__memanalysis G__P(());
int G__memresult G__P(());
void G__DUMMY_Free G__P((void* p));
void *G__TEST_fopen G__P((char *fname,char *mode));
int G__TEST_fclose G__P((FILE* p));
void *G__TEST_tmpfile G__P(());

#define malloc(x) G__TEST_Malloc(x)
#define calloc(x,y) G__TEST_Calloc(x,y)
#define free(x) G__TEST_Free(x)
#define realloc(p,x) G__TEST_Realloc(p,x)
#define fopen(x,y) G__TEST_fopen(x,y)
#define fclose(x) G__TEST_fclose(x)
#define tmpfile G__TEST_tmpfile

#endif /* G__MEMTEST_H */


/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
