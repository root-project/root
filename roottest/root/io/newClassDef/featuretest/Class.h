#ifndef CLASS_H
#define CLASS_H

class MyClass {
protected:
  bool GetProtected();
public:
  bool GetPublic() { return GetProtected(); }
};

#endif
