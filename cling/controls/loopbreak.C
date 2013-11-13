void loopbreak() {
  Int_t i=0;
  while (i++<100) {
     printf ("in loop, i=%d\n",i);
     break;
  }
  fprintf (stderr,"after loop, i=%d\n",i);
  // QWERTYUI;
}
