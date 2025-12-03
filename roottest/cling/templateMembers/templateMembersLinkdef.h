// found
#pragma link C++ class Base+;
#pragma link C++ class Derived+;
#pragma link C++ class TemplateClass<Derived>+;

#pragma link C++ class my_shared_ptr<Base>+;
#pragma link C++ class my_shared_ptr<Derived>+;

#pragma link C++ function my_shared_ptr<Base>::f1(Derived)+;
#pragma link C++ function my_shared_ptr<Base>::f2<Derived>()+;
#pragma link C++ function my_shared_ptr<Base>::f3(TemplateClass<Derived>)+;
#pragma link C++ function my_shared_ptr<Base>::f4(TemplateClass<Derived>)+;
#pragma link C++ function my_shared_ptr<Base>::f5(const my_shared_ptr<Derived>&)+;
#pragma link C++ function my_shared_ptr<Base>::f6(const my_shared_ptr<Derived>&)+;
#pragma link C++ function my_shared_ptr<Base>::f7(const my_shared_ptr<Derived>&)+;
#pragma link C++ typedef my_shared_ptr<Base>::reference;
#pragma link C++ function my_shared_ptr<Base>::f8(const my_shared_ptr<Derived>&)+;
#pragma link C++ function my_shared_ptr<Base>::f8(const my_shared_ptr<Base>&)+;

#pragma link C++ function my_shared_ptr<Base>::n1(const my_shared_ptr<Base>&);
#pragma link C++ function my_shared_ptr<Base>::n2(const my_shared_ptr<Base>&);
#pragma link C++ function my_shared_ptr<Base>::n3(const my_shared_ptr<Base>&);
#pragma link C++ function my_shared_ptr<Base>::n4(const my_shared_ptr<Base>&);

#pragma link C++ function my_shared_ptr<Base>::n1(const my_shared_ptr<Derived>&);
#pragma link C++ function my_shared_ptr<Base>::n2(const my_shared_ptr<Derived>&);
#pragma link C++ function my_shared_ptr<Base>::n3(const my_shared_ptr<Derived>&);
#pragma link C++ function my_shared_ptr<Base>::n4(const my_shared_ptr<Derived>&);

#pragma link C++ function my_shared_ptr<Base>::operator=(const my_shared_ptr<Derived>&)+;

#pragma link C++ function my_shared_ptr<Base>::my_shared_ptr<Base>(const my_shared_ptr<Derived>&);



