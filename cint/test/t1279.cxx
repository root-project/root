/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <iostream>

class MyClass
{
public:
	int i;
};

template < typename T >
T *Pass( T *t )
{
   return t;
}

int main()
{
	MyClass *ptr = new MyClass;

        // Works in CINT & g++
	ptr->i = 10;
	std::cout << ( ptr )->i << std::endl;
	std::cout << Pass< MyClass >( ptr )->i << std::endl;

        // Works in CINT & g++
	MyClass *new_ptr = Pass< MyClass >( ptr );
        new_ptr->i = 20; 
	std::cout << ( new_ptr )->i << std::endl;
	std::cout << Pass< MyClass >( new_ptr )->i << std::endl;

        // Works in CINT & g++	
	(Pass< MyClass >( ptr ))->i = 30;
	std::cout << ( ptr )->i << std::endl;
	std::cout << Pass< MyClass >( ptr )->i << std::endl;

	// Used to not work in CINT
	Pass< MyClass >( ptr )->i = 40;
	std::cout << ( ptr )->i << std::endl;
	std::cout << Pass< MyClass >( ptr )->i << std::endl;

	// Works in CINT (note the white space)
        (Pass< MyClass >( ptr ) )->i = 50;
        std::cout << Pass< MyClass >( ptr )->i << std::endl;

	// Does not work in CINT (note the white space)
        ( Pass< MyClass >( ptr ))->i = 50;
        std::cout << Pass< MyClass >( ptr )->i << std::endl;


        // Does not work in CINT (note the white space)
        ( Pass< MyClass >( ptr ) )->i = 50;
        std::cout << Pass< MyClass >( ptr )->i << std::endl;

	return 0;
}
