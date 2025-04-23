#define pass(x) x
#define pass(x,y) x,y

void f() {
  pass(vector<int>);
  pass(map<int,double>);
}
