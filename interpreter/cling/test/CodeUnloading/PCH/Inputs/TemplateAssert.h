#ifndef TemplateAssert_h
#define TemplateAssert_h

template <typename T> struct Trait {
  static constexpr bool Value = false;
};

template <> struct Trait<double> {
  static constexpr bool Value = true;
};

template <typename T> struct RequiresTrait {
  void Scale(double) { static_assert(Trait<T>::Value, "invalid type"); }
};

void inst() { RequiresTrait<int> r; }

#endif
