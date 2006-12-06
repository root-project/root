/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

int main()
{
  unsigned long ulCRC;
  int i;
  
  for(i=0;i<3;i++) {
    ulCRC = 0xA6E0A697;
    ulCRC /= 256;
    printf("ulCRC = 0x%lX\n", ulCRC);
  }

  for(i=0;i<3;i++) {
    ulCRC = 0x86E0A697;
    ulCRC *= 2;
    printf("ulCRC = 0x%lX\n", ulCRC);
  }
  for(i=0;i<3;i++) {
    ulCRC = 0xA6E0A697;
    ulCRC += 2;
    printf("ulCRC = 0x%lX\n", ulCRC);
  }
  for(i=0;i<3;i++) {
    ulCRC = 0xA6E0A697;
    ulCRC -= 2;
    printf("ulCRC = 0x%lX\n", ulCRC);
  }
  for(i=0;i<3;i++) {
    ulCRC = 0xA6E0A697;
    ulCRC %= 256;
    printf("ulCRC = 0x%lX\n", ulCRC);
  }
  return 0;
}

