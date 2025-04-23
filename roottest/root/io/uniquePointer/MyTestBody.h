//class MyTest{

private:
	template<class COLL>
	std::string CollToString(const COLL& coll) const {
		std::string s;
		for (auto& v : coll) s+= v->ToString();
		return s;
	}
	
	template<class COLL>
	std::string CollOfCollToString(const COLL& collOfColl) const {
		std::string s;
		for (auto& coll : collOfColl) {
	    	for (auto& v : coll) s+= v->ToString();
		}
		return s;
	}

	template<class MAP>
	std::string MapToString(const MAP& map) const {
		std::string s;
		for (auto& v : map) s+= std::to_string(v.first) + v.second->ToString();
		return s;
	}
#ifdef USE_UNIQUE
	std::vector<std::unique_ptr<MyObj>> m0;
	std::list<std::unique_ptr<MyObj>> m1;
	std::forward_list<std::unique_ptr<MyObj>> m2;
	std::map<float, std::unique_ptr<MyObj>> m3;
	std::vector<std::vector<std::unique_ptr<MyObj>>> m4;
        std::unordered_map<float, std::unique_ptr<MyObj>> m5;
#else
	std::vector<MyObj*> m0;
	std::list<MyObj*> m1;
	std::forward_list<MyObj*> m2;
	std::map<float, MyObj*> m3;
	std::vector<std::vector<MyObj*>> m4;
        std::unordered_map<float, MyObj*> m5;
#endif
public:
	MyTest(int init=0) {
		for (int i =init ; i < init+101; i++) {
			m0.emplace_back(new MyObj(i));
			m1.emplace_back(new MyObj(i+1));
			m2.emplace_front(new MyObj(i+2));
#ifdef USE_UNIQUE
			m3.emplace(std::piecewise_construct,
                       std::forward_as_tuple(i),
                       std::forward_as_tuple(std::unique_ptr<MyObj>(new MyObj(i+3))));
			m5.emplace(std::piecewise_construct,
                       std::forward_as_tuple(i),
                       std::forward_as_tuple(std::unique_ptr<MyObj>(new MyObj(i+5))));
#else
			m3.emplace(std::make_pair(i,new MyObj(i+3)));
                        m5.emplace(std::make_pair(i,new MyObj(i+5)));
#endif

			decltype(m0) m4_element;
			for (int j = 0; j < 3 ; ++j) {
				m4_element.emplace_back(new MyObj(i+4));
			}
			m4.emplace_back(std::move(m4_element));
		}
	}
	std::string ToString() const {
		std::string s;
		s += CollToString(m0);
		s += CollToString(m1);
		s += CollToString(m2);
		s += MapToString(m3);
		s += CollOfCollToString(m4);
                //s += CollOfCollToString(m5); // unordered!
		// We test the writing/reading of unordered w/o content
		return s;
	}
//};
