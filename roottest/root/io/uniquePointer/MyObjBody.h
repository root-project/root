//class MyObj{
public:
	MyObj(int init = 0):fVal(init){
		for (int i = init; i < init + 4; ++i) {
			fVect.emplace_back(i);
		}
	}
	std::string ToString() const { 
		auto s =  std::to_string(fVal);
		for (auto&& v : fVect) s += std::to_string(v);
		return s;
	}
private:
	int fVal;
	vector<double> fVect;
//};