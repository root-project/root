// $Id: sigsegv.h,v 1.3 2009-12-16 17:17:38 avalassi Exp $
// From http://tlug.up.ac.za/wiki/index.php/Obtaining_a_stack_trace_in_C_upon_SIGSEGV
#ifndef SIGSEGV_H
#define SIGSEGV_H 1
#ifdef __cplusplus
//extern "C" // AV
#endif
namespace cool
{
  int setup_sigsegv();
}
#endif
