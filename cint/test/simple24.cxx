class Foo {
public:
  Foo(int i) { 
    printf("foo-ctor: %d\n",i);
  }
  int *pointer() {return 0;}
};

int main()
{
  Foo f1=2;
  return 0;
}
