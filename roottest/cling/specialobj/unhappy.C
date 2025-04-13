// Checks unknown identifers don't crash ROOT (ROOT-8323).

class A {
public:
  static int f(){ I_really_do_not_exist(); return 0; }
};

int i = A::f();
