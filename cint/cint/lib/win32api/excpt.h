#if _MSC_VER > 1000 
#pragma once 
#endif 

#ifndef _INC_EXCPT 
#define _INC_EXCPT 
#include <crtdefs.h> 
#ifdef  _MSC_VER 
#pragma pack(push,_CRT_PACKING) 
#endif 

#ifdef  __cplusplus 
extern "C" { 
#endif 
typedef enum _EXCEPTION_DISPOSITION { 
 ExceptionContinueExecution, 
 ExceptionContinueSearch, 
 ExceptionNestedException, 
 ExceptionCollidedUnwind 
} EXCEPTION_DISPOSITION; 
#ifdef  _MSC_VER 

#define GetExceptionCode _exception_code 
#define exception_code _exception_code 
#define GetExceptionInformation (struct _EXCEPTION_POINTERS *)_exception_info 
#define exception_info (struct _EXCEPTION_POINTERS *)_exception_info 
#define AbnormalTermination _abnormal_termination 
#define abnormal_termination _abnormal_termination 

unsigned long __cdecl _exception_code(void); 
void * __cdecl _exception_info(void); 
int __cdecl _abnormal_termination(void); 

#endif 

#define EXCEPTION_EXECUTE_HANDLER       1 
#define EXCEPTION_CONTINUE_SEARCH       0 
#define EXCEPTION_CONTINUE_EXECUTION    -1 

#ifdef  __cplusplus 
} 
#endif 

#ifdef  _MSC_VER 
#pragma pack(pop) 
#endif  /* _MSC_VER */ 

#endif  /* _INC_EXCPT */ 

