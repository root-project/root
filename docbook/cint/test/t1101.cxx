/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <cstdio>
#include <cstring>

using namespace std;

// badinit.c
const char* system = "WinNT";

//______________________________________________________________________________
void ls(const char* path = 0)
{
   // --
   char s[256];
   s[0] = '\0';
#ifdef __CINT__
   if (!strcmp(system, "WinNT")) {
      strcpy(s, "dir /w ");
   }
   else {
      strcpy(s, "ls ");
   }
#else // __CINT__
   strcpy(s, "dir /w ");
#endif // __CINT__
   if (path) {
      strcat(s, path);
   }
   //gSystem.Exec(s);
   printf("%s\n", s);
}

int main()
{
   for (int i = 0; i < 2; ++i) {
      ls();
   }
   return 0;
}

