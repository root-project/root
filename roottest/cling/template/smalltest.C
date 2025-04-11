template <class T> class Array {

};
class Buffer {};
#define AFTER_FIRST_FIX
#ifndef AFTER_FIRST_FIX
template <class Tmpl> Buffer& operator>>(Buffer &buf, Tmpl *&obj) {
  fprintf(stderr,"Generic operator>>\n");
  return buf;
}

template <class T> Buffer& operator>>(Buffer &buf, Array<T> *&obj) {
  fprintf(stderr,"Array operator>>\n");
  return buf;
}
#elif defined(__CINT__)
template <class Tmpl> Buffer &operator>>(Buffer &buf, Tmpl *&obj) {
  fprintf(stderr,"Generic operator>>\n");
  return buf;
}

template <class T> Buffer &operator>>(Buffer &buf, Array<T> *&obj) {
  fprintf(stderr,"Array operator>>\n");
  return buf;
}
#else
template <class Tmpl> Buffer &operator>>(Buffer &buf, Tmpl *&) {
  fprintf(stderr,"Generic operator>>\n");
  return buf;
}

template <class T> Buffer &operator>>(Buffer &buf, Array<T> *&) {
  fprintf(stderr,"Array operator>>\n");
  return buf;
}
#endif

void test() {
  Buffer b;
  int *i;
  Array<int> *a;
  
  b >> i;
  b >> a;

}

void smalltest() { test(); }
