void Unsupported(const std::string &what) {
   std::cerr << "ROOT " << ROOT_RELEASE << " does not support "
             << what << std::endl;
}

void TestError(const std::string &test, const char *msg) {
   std::cerr << "Error for '" << test << "' : " << msg << "\n";
}
void TestError(const std::string &test, const std::string &str) {
   TestError(test, str.c_str());
}

template <class T> void TestError(const std::string &test, 
                                  const std::vector<T> &orig, 
                                  const std::vector<T> &copy) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class T> void TestError(const std::string &test, 
                                  std::vector<T> *orig, 
                                  std::vector<T> *copy) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

void TestError(const std::string &test, const Helper &orig, const Helper &copy) {
   TestError(test, Form("Helper object wrote %d and read %d\n",
                        orig.val,copy.val));
}

void TestError(const std::string &test, Helper* orig, Helper* copy) {
   if (orig==0 || copy==0) {
      TestError(test,Form("For Helper, non-initialized pointer %p %p",orig,copy));
   } else {
      TestError(test, *orig, *copy); 
   }
}

template <class T> void TestError(const std::string &test, const T &orig, const T &copy) {
   std::stringstream s;
   s << "We wrote: " << orig << " but read " << copy << std::ends;
   TestError(test, s.str());
}


