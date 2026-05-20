#ifndef DEQUE_HOLDER_H
#define DEQUE_HOLDER_H

#ifdef TEST_CONT
#undef TEST_CONT
#undef TEST_CONT_HOLDER 
#endif

#define TEST_CONT std::deque
#define TEST_CONT_HOLDER dequeHolder

#include "contHolder.h"
#endif
