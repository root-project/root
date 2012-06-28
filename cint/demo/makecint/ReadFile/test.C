/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

main()
{
  ReadFile a=ReadFile("ReadFile.h");

  a.setendofline("#");
  a.setseparator(" \t.()");

  int i;
  while(a.read()) {
    printf("argc=%d :",a.argc);
    for(i=1;i<=a.argc;i++) printf("%s | ",a.argv[i]);
    putchar('\n');
  }
}
