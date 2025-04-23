void f() try {
  std::cout << "f()" << std::endl;
  throw 1;
} catch (...) {
  std::cout << "caught integer!" << std::endl;
}

void trybody() {
  f();
}
