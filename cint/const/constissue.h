int get_histogram(char const* , char const* ){return 0; }
template<class T> void get_histogram(T const & t, char const * name);
#ifdef __CINT__
// Can't remember what this is actually testing but it should do
// any harm anyway!
#pragma link C++ function get_histogram;
#endif

#include <map>

class TriggerCharacterizer {
 public:
  TriggerCharacterizer() {}
 private:
  typedef std::map<char const*, unsigned long> trigger_counting_map;

  trigger_counting_map m_map; 
};

#ifdef __CINT__
//#pragma link C++ class TriggerCharacterizer::trigger_counting_map;
#endif

void constissue() {}
