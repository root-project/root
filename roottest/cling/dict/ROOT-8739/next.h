namespace std {
  template <typename T> void Next(T*);
}

namespace Functions {
  void OtherNext();
}

using namespace Functions;

namespace next {
   class Inside_next {};
}

namespace Next {
  class Inside_Next {};
}

namespace OtherNext {
  class Inside_OtherNext {};
}

namespace YetAnotherNext {
   class Inside_YetAnotherNext {};
}
