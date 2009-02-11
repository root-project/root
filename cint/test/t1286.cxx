#include <string> 
namespace std {} using namespace std; 

class B 
{ 
public: 
   string str; 
}; 

int t1286() 
{ 
   B * arrB = new B[10];
   for(int i=0;i<10;++i) {
      arrB[i].str = "test";
      arrB[i].str += ( 'a'+i );
   }
   for(int i=0;i<10;++i) {
      printf(arrB[i].str.c_str());
      printf("\n");
   }

   if( arrB ) 
      delete[] arrB; 
   return 0;
}

int main() {
   return t1286();
}
