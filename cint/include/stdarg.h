/****************************************************************
* stdarg.h
*****************************************************************/
#ifndef G__STDARG_H
#define G__STDARG_H

typedef double *va_list;

void va_start(va_list listptr,int dummy)
{
	fprintf(stderr,"Limitation: va_start(),va_arg(),va_end() not supported\n");
}

#endif
