{
   TheClass t;
// printf("%d", t.a(0.)); ambigous a(int, int) a(float, int)
   printf("%d", t.a((char)0));
   printf("%d", t.a('a'));
   printf("%d", t.a("a"));
   float arrf[3];
   printf("%d", t.a(arrf));

   int i;
   printf("%d", t.a(0,&i));
   printf("%d", t.a(-1,3));
   printf("%d", t.a(0.,1.f));
   printf("%d", t.a(-1,'a'));
   printf("\n");
}
