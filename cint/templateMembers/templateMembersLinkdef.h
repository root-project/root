// found
#pragma link C++ class Base+;
#pragma link C++ class Derived+;
#pragma link C++ class TemplateClass<Derived>+;

#pragma link C++ class shared_ptr<Base>+;
#pragma link C++ class shared_ptr<Derived>+;

#pragma link C++ function shared_ptr<Base>::f1(Derived)+;
#pragma link C++ function shared_ptr<Base>::f2<Derived>()+;
#pragma link C++ function shared_ptr<Base>::f3(TemplateClass<Derived>)+;
#pragma link C++ function shared_ptr<Base>::f4(TemplateClass<Derived>)+;
#pragma link C++ function shared_ptr<Base>::f5(const shared_ptr<Derived>&)+;
#pragma link C++ function shared_ptr<Base>::f6(const shared_ptr<Derived>&)+;
#pragma link C++ function shared_ptr<Base>::f7(const shared_ptr<Derived>&)+;
#pragma link C++ typedef shared_ptr<Base>::reference;
#pragma link C++ function shared_ptr<Base>::f8(const shared_ptr<Derived>&)+;
#pragma link C++ function shared_ptr<Base>::f8(const shared_ptr<Base>&)+;

#pragma link C++ function shared_ptr<Base>::n1(const shared_ptr<Base>&);
#pragma link C++ function shared_ptr<Base>::n2(const shared_ptr<Base>&);
#pragma link C++ function shared_ptr<Base>::n3(const shared_ptr<Base>&);
#pragma link C++ function shared_ptr<Base>::n4(const shared_ptr<Base>&);

#pragma link C++ function shared_ptr<Base>::n1(const shared_ptr<Derived>&);
#pragma link C++ function shared_ptr<Base>::n2(const shared_ptr<Derived>&);
#pragma link C++ function shared_ptr<Base>::n3(const shared_ptr<Derived>&);
#pragma link C++ function shared_ptr<Base>::n4(const shared_ptr<Derived>&);

#pragma link C++ function shared_ptr<Base>::operator=(const shared_ptr<Derived>&)+;

#pragma link C++ function shared_ptr<Base>::shared_ptr<Base>(const shared_ptr<Derived>&);



