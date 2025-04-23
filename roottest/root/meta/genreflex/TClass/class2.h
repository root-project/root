#ifndef CLASS2_H
#define CLASS2_H

#include <string>
#include <map>
#include <vector>
#include <set>
#include <queue>
#include <deque>
#include <list>

class class2 {

public:
   class2(int i, int j);
   class2();
   ~class2();
   int getI();
   void setI(int i);
   int getJ();
   void setJ(int i);
private:
   int m_i;
   int m_j;
   // Strings
//    std::string m_string;
   // Various vectors
//    std::vector<std::string> m_vector1;
//    std::vector<int> m_vector2;
//    std::vector <std::pair<int,double> > m_vector3;
//    std::vector <std::pair<int,std::string> > m_vector4;
//    std::vector <std::pair <float, std::pair<int,double> > > m_vector5;
   // Maps - multimaps
   std::map <float,double> m_map1;
//    std::map <float,std::string> m_map2;
//    std::multimap <float,double> m_mmap1;
//    std::multimap <float,std::string> m_mmap2;
//    // queue,deque, set, multiset
//    std::queue<unsigned int> m_queue1;
//    std::deque<unsigned int> m_deque1;
//    std::set<float> m_set1;
//    std::multiset<float> m_mset1;
//    // list
//    std::list<double> m_list1;
};
#endif


class2::class2(int i, int j):
  m_i(i),
  m_j(j){};

class2::class2():
   m_i(0),
   m_j(0){};

class2::~class2(){};

int class2::getI(){return m_i;};

void class2::setI(int i){m_i=i;};

int class2::getJ(){return m_j;};

void class2::setJ(int j){m_j=j;};
