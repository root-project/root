#ifndef LIST_HOLDER_H
#define LIST_HOLDER_H

#ifdef TEST_CONT
#undef TEST_CONT
#undef TEST_CONT_HOLDER 
#endif

#define TEST_CONT std::list
#define TEST_CONT_HOLDER listHolder

#include "contHolder.h"
#endif

