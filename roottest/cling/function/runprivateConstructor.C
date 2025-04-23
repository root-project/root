#undef private
#undef protected
class top {
private:
   top () {}
public:
   top(int i) {}
};
class bottom : public top {
private:
   bottom() {}
};

void runprivateConstructor() {
   bottom *b = new bottom;
}
