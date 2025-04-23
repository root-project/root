class master {
 public:
  master() {};
  ~master() {};

  template <class obj> int GetValue(const obj &o) { return o.GetValue(); }

};
