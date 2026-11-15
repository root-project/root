// Instantiations of the call operator of generic lambdas are function-local
// declarations that are passed to the consumer before the enclosing function.
inline int genericLambda() { return [](auto x) { return x; }(1); }

inline int genericLambdaRetType() {
  return [](auto x) -> int { return x; }(2);
}

inline auto makeGenericLambda() {
  return [](auto x) { return x; };
}
