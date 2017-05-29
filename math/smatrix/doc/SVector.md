\page SVectorDoc SVector Class Properties

The template ROOT::Math::SVector class has 2 template parameters which define, at compile time, its properties. These are:

*   type of the contained elements, for example _float_ or double.
*   size of the vector.

### Creating a Vector

The following constructors are available to create a vector:

*   Default constructor for a zero vector (all elements equal to zero)
*   Constructor (and assignment) from a vector expression, like v = p*q + w. Due to the expression template technique, no temporary objects are created in this operation.
*   Construct a vector passing directly the elements. This is possible only for vector up to size 10\.
*   Constructor from an iterator copying the data refered by the iterator. It is possible to specify the _begin_ and _end_ of the iterator or the _begin_ and the size. Note that for the Vector the iterator is not generic and must be of type _T*,_ where T is the type of the contained elements.

Here are some examples on how to create a vector. In the following we assume that we are using the namespace ROOT::Math.

~~~ {.cpp}
SVector>double,N>  v;                         _// create a vector of size N, v[i]=0_
SVector>double,3>  v(1,2,3);                  _// create a vector of size 3, v[0]=1,v[1]=2,v[2]=3_
double   a[9] = {1,2,3,4,5,6,7,8,9};          _// input data_
SVector>double,9>  v(a,9);                    _// create a vector using the a[] data_
~~~


### Accessing and Setting Methods

The single vector elements can be set or retrieved using the _operator[i]_ , _operator(i)_ or the iterator interface. Notice that the index starts from zero and not from one as in FORTRAN. Also no check is performed on the passed index. Furthermore, all the matrix elements can be set also by using the ROOT::SVector::SetElements function passing a generic iterator. The elements can be accessed also by using the ROOT::Math::SVector::apply(i) function.

~~~ {.cpp}
v[0]  = 1;                          _ // set the first element _
v(1)  = 2;                          _ // set the second element _
*(v.**begin**()+3) = 3; _// set the third element_
_// set vector elements from a std::vector<double>::iterator</double>_
std::vector <double>w(3);
v.SetElements(w.begin(),w.end());

double x = m(i);                     _// return the i-th element_
x = m.**apply**(i);                      _// return the i-th element_
x = *(m.**begin**()+i);                  _// return the i-th element
~~~


In addition there are methods to place a sub-vector in a vector. If the size of the the sub-vector is larger than the vector size a static assert ( a compilation error) is produced.

~~~ {.cpp}
SVector>double,N>  v;
SVector>double,M>  w;          _// M <= N otherwise a compilation error is obtained later _
_// place a vector of size M starting from element ioff,  v[ioff + i] = w[i]_
v.**Place_at**(w,ioff);
_// return a sub-vector of size M starting from v[ioff]:  w[i] = v[ioff + i]_
w = v.Sub < SVector>double,M> > (ioff);
~~~


For additional Vector functionality see the \ref MatVecFunctions page
