#ifndef ROOT_strtok
#define ROOT_strtok

#include <string.h>

// On Windows strtok_r does not exist: the function is called strtok_s.
inline char *R__STRTOK_R(char *str, const char *delim, char **saveptr)
{
#if defined(_WIN32)
   return strtok_s(str, delim, saveptr);
#else
   return strtok_r(str, delim, saveptr);
#endif
}

#endif
