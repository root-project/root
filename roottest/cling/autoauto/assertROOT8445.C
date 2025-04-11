// Test for ROOT-8445
int assertROOT8445() {
  struct Foo {
    struct Bar {
      enum Baz {
        kOne,
        kTwo,
        kThree
      };
    };
  };
  auto var = Foo::Bar::kThree;
  return 0;
}
