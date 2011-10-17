int fff(int i) { return i+1; }

int pp() {
#define OBSCURE(o) f##o(o)
#define HIDE(h) OBSCURE(h##h)
#define CRYPT(c) HIDE(c) * HIDE(c)
   int ff = 2;
   printf("%d\n", CRYPT(f));
}
