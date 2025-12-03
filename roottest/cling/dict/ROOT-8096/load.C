template <class Anchor, typename ... T> struct Packing;

int load() {
  Packing<float,int,Packing<int, double>> p;
  return 0;
}
