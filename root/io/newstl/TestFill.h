template <class T> void fill(T& filled, UInt_t seed);

template <> void fill(Helper& filled, UInt_t seed) {
   filled.val = seed;
}

template <> void fill(THelper& filled, UInt_t seed) {
   filled.val = seed;
}

template <> void fill(double& filled, UInt_t seed) {
   filled = seed;
}

template <> void fill(float& filled, UInt_t seed) {
   filled = seed;
}

template <> void fill(int& filled, UInt_t seed) {
   filled = seed;
}

template <> void fill(short& filled, UInt_t seed) {
   filled = seed;
}

template <> void fill(char& filled, UInt_t seed) {
   filled = seed;
}

template <> void fill(TString& filled, UInt_t seed) {
   UInt_t size = seed%20;
   filled = "";
   for(UInt_t i=0; i<size; i++) {
      filled += ((i&seed)+'a');
   }
}

template <> void fill(std::string& filled, UInt_t seed) {
   TString temp; 
   fill(temp,seed);
   const_cast<std::string&>(filled) = temp.Data();
}

template <> void fill(const std::string& filled, UInt_t seed) {
   TString temp; 
   fill(temp,seed);
   const_cast<std::string&>(filled) = temp.Data();
}

template <> void fill(TNamed& filled, UInt_t seed) {
   TString temp; 
   fill(temp,seed);
   filled.SetName(temp.Data());
   fill(temp,seed+1);
   filled.SetTitle(temp.Data());
}

template <> void fill(const TNamed& cfilled, UInt_t seed) {
   TNamed &filled = const_cast<TNamed&>(cfilled);
   fill(filled,seed);
}
const char* GetEHelperStringValue(const EHelper &eval);
template <> void fill(EHelper& filled, UInt_t seed) {
   switch ( seed%3 ) {
      case 0: filled = kZero; break;
      case 1: filled = kOne;  break;
      case 2: filled = kTwo;  break;
      default: filled = kEnd; break;
   }
}

template <class T> void fill(T& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      typename T::value_type val;
      fill(val,seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fill(std::vector<T*>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val = new T;
      //      fprintf(stderr,"trying to fill vector of %s* with %p\n",
      //        typeid(T).name(),val);
      fill(*val,seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fill(std::deque<T*>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val = new T;
      fill(*val,seed*10+i);
      filled.push_back(val);
  }  
}
