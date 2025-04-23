void textcopy() {
   printf("Making wchar text...\n");
   TText widefoo(0, 0, L"wide-fooä");
   printf("copying...\n");
   TText widebar;
   widefoo.Copy(widebar);
   printf("good!\n");

   printf("Making normal char text...\n");
   TText foo(0, 0, "foo");
   printf("copying...\n");
   TText bar;
        foo.Copy(bar);
   printf("good!\n");
}
