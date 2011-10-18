// We would expect both namespaces to be reverted from the AST, however when
// clang tries to remove the non-canonical declaration it removes the canonical
// one instead.
namespace test {
  int i;
}

namespace test {
  int i; // expected-error {{redefinition of 'i'}} expected-note {{previous definition is here}} 
}
