#if !defined		(_ACCESS_SPECIFIERS_HPP__)
#define				 _ACCESS_SPECIFIERS_HPP__

/**
 * @class X
 *
 * See 11.0.2 [class.access]
 * Members of a class defined with the keyword <code>class</code> are <code>private</code> by default.
 * Members of a class defined with the keywords <code>struct</code> or <code>union</code> are <code>public</code> by default.
 *
 */
class X {
	int a;				// X::a is private by default
};

struct S {
	int a;				// S::a is public by default
};


#endif			   //_ACCESS_SPECIFIERS_HPP__
