class myclass {};

void func(const char*, const myclass&) {}
void func(const myclass&, const char*) {}
void func(const myclass&, const myclass&) {}

void main() {
   myclass a;
   myclass b;
   func(a,b);
}
