template <class T>
bool IsSame(const T& a, const T& b){
   cout << "ERROR\n";
   return a==b;
}

template <>
bool IsSame<>(const double& a, const double& b){
   if (a==b) return true;
   cout << "Error numbers differ: " << a << " " << b << std::endl;
   return false;
}

template<class Cont>
bool IsSameCont(const Cont& a, const Cont& b){
   auto size =std::distance(a.begin(),a.end());
   if (size != std::distance(b.begin(),b.end())) return false;
   for (auto aIt = a.cbegin(), bIt = b.begin();aIt!=a.end();aIt++,bIt++ ){
      if (! IsSame(*aIt, *bIt)) return false;
   }

   return true;
}

template <class T, class ALLOCATOR>
bool IsSame(const std::forward_list<T,ALLOCATOR>& a, const std::forward_list<T,ALLOCATOR>& b){
   return IsSameCont(a,b);
}

template <class T, class ALLOCATOR>
bool IsSame(const std::list<T,ALLOCATOR>& a, const std::list<T,ALLOCATOR>& b){
   return IsSameCont(a,b);
}
template <class T, class ALLOCATOR>
bool IsSame(const std::vector<T,ALLOCATOR>& a, const std::vector<T,ALLOCATOR>& b){
   return IsSameCont(a,b);
}
template <class T, class ALLOCATOR>
bool IsSame(const std::deque<T,ALLOCATOR>& a, const std::deque<T,ALLOCATOR>& b){
   return IsSameCont(a,b);
}

template <>
bool IsSame<>(const TH1F& a, const TH1F& b){
   if( 0 != strcmp(a.GetName(),b.GetName())) return false;
   if( 0 != strcmp(a.GetTitle(),b.GetTitle())) return false;
   if( a.GetNbinsX() != b.GetNbinsX()) return false;
   for (size_t i=0;i<a.GetNbinsX();++i){
      if (a.GetBinContent(i)!=b.GetBinContent(i)) return false;
      if (a.GetBinError(i)!=b.GetBinError(i)) return false;
   }
   return true;
}

void createFile(const char* filename){
   auto file = TFile::Open(filename,"RECREATE");
   delete file;
}

template<class T>
void writeToFile(const T& obj, const char* objName, const char* filename){
   auto file = TFile::Open(filename,"UPDATE");
   file->WriteObject(&obj,objName);
   delete file;
}

template<class T>
void readAndCheckFromFile(const T& obj, const char* objName, const char* filename){
   auto file = TFile::Open(filename,"READ");
   auto objFromFile =(T*)file->Get(objName);
   if (!objFromFile){
      std::cerr << "Error in reading object " << objName << " from file " << filename << "\n";
      delete file;
      return;
   }
   if (!IsSame(obj,*objFromFile)) {
      std::cerr << "Error: object " << objName << " read from file " << filename << " from file and in memory are not identical!\n";
   }

   delete file;
}

template<class T>
void writeReadCheck(const T& obj, const char* objName, const char* filename){
   writeToFile(obj,objName,filename);
   readAndCheckFromFile(obj,objName,filename);
}

template<class Cont>
void fillHistoCont(Cont& cont, unsigned int n=5000){
   for (auto& h:cont) h.FillRandom("gaus",n);
}
template<class NestedCont>
void fillHistoNestedCont(NestedCont& nestedCont, unsigned int n=5000){
   for (auto& hCont:nestedCont) {
      for (auto& h:hCont){
         h.FillRandom("gaus",n);
      }
   }
}

template<class Cont>
void randomizeCont(Cont& cont){
   for (auto& el : cont){
      el*=gRandom->Uniform(1,2);
   }
}
