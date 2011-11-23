// RUN: cat %s | %cling -Xclang -verify

class Klass{};
Klass s;
Klass s; // expected-error {{redefinition of 's'}} expected-note {{previous definition is here}}

const char* a = "test";
const char* a = ""; // expected-error {{redefinition of 'a'}} expected-note {{previous definition is here}}

.q
