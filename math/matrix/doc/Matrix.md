/**


\page MatrixPage The ROOT Matrix Linear Algebra classes.


The ROOT linear algebra package provides a complete environment in ROOT to perform matrix calculations such as matrix-vector and matrix-matrix  multiplications and other linear algebra calculations like equation solving and eigenvalue decompositions.

The present package implements all the basic algorithms dealing
 with vectors, matrices, matrix columns, rows, diagonals, etc.
 In addition eigen-Vector analysis and several matrix decomposition
 have been added (LU,QRH,Cholesky,Bunch-Kaufman and SVD) .
 The decompositions are used in matrix inversion, equation solving.


### Matrix classes

ROOT provides the following matrix classes, among others:

- `TMatrixDBase`

- `TMatrixF`

- `TMatrixFSym`

- `TVectorF`

- `TMatrixD`

- `TMatrixDSym`

- `TMatrixDSparse`

- `TDecompBase`

- `TDecompChol`

For a dense matrix, elements are arranged in memory in a ROW-wise
 fashion . For (n x m) matrices where n*m <=kSizeMax (=25 currently)
 storage space is available on the stack, thus avoiding expensive
 allocation/deallocation of heap space . However, this introduces of
 course kSizeMax overhead for each matrix object . If this is an
 issue recompile with a new appropriate value (>=0) for kSizeMax

 Sparse matrices are also stored in row-wise fashion but additional
 row/column information is stored, see TMatrixTSparse source for
 additional details .

 Another way to assign and store matrix data is through Use
 see for instance stressLinear.cxx file .

 Unless otherwise specified, matrix and vector indices always start
 with 0, spanning up to the specified limit-1. However, there are
 constructors to which one can specify aribtrary lower and upper
 bounds, e.g. TMatrixD m(1,10,1,5) defines a matrix that ranges
 from 1..10, 1..5 (a(1,1)..a(10,5)).

#### Matrix properties

A matrix has five properties, which are all set in the constructor:

- `precision` <br>
If the `precision` is float (i.e. single precision), use the `TMatrixF` class family. If the precision is double, use the `TMatrixD` class family.

- `type`<br>
Possible values are: `general` (`TMatrixD`), `symmetric` (`TMatrixDSym`) or `sparse` (`TMatrixDSparse`).

- `size`<br>
Number of rows and columns.

- `index`<br>
Range start of row and column index. By default these start at 0.

- `sparse map`<br>
Only relevant for a sparse matrix. It indicates where elements are unequal 0.

#### Accessing matrix properties

Use one of the following methods to access the information about the relevant matrix property:

- `Int_t GetRowLwb()`: Row lower-bound index.

- `Int_t GetRowUpb()`: Row upper-bound index.

- `Int_t GetNrows()`: Number of rows.

- `Int_t GetColLwb()`: Column lower-bound index.

- `Int_t GetColUpb()`: Column upper-bound index.

- `Int_t GetNcols()`: Number of columns.

- `Int_t GetNoElements()`: Number of elements, for a dense matrix this equals: `fNrows x fNcols`.

- `Double_t GetTol()`: Tolerance number that is used in decomposition operations.

- `Int_t *GetRowIndexArray()`: For sparse matrices, access to the row index of `fNrows+1` entries.

- `Int_t *GetColIndexArray()`: For sparse matrices, access to the column index of `fNelems` entries.

#### Setting matrix properties

Use one of the following methods to set a matrix property:

- `SetTol (Double_t tol)`<br>
Sets the tolerance number.

- `ResizeTo (Int_t nrows,Int_t ncols, Int_t nr_nonzeros=-1)`<br>
Changes the matrix shape to `nrows x ncols`. Index will start at 0.

- `ResizeTo(Int_t row_lwb,Int_t row_upb, Int_t col_lwb,Int_t col_upb, Int_t nr_nonzeros=-1)`<br>
Changes the matrix shape to `row_lwb:row_upb x col_lwb:col_upb`.

- `SetRowIndexArray (Int_t *data)`<br>
For sparse matrices, it sets the row index. The array data should contain at least `fNrows+1` entries column lower-bound index.

- `SetColIndexArray (Int_t *data)`<br>
For sparse matrices, it sets the column index. The array data should contain at least `fNelems` entries.

- `SetSparseIndex (Int_t nelems new)`<br>
Allocates memory for a sparse map of `nelems_new` elements and copies (if exists) at most `nelems_new` matrix elements over to the new structure.

- `SetSparseIndex (const TMatrixDBase &a)`<br>
Copies the sparse map from matrix `a`.

- `SetSparseIndexAB (const TMatrixDSparse &a, const TMatrixDSparse &b)`<br>
Sets the sparse map to the same map of matrix `a` and `b`.

### Creating and filling a matrix

Use one of the following constructors to create a matrix:

- `TMatrixD(Int_t nrows,Int_t ncols)`
- `TMatrixD(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)`
- `TMatrixD(Int_t nrows,Int_t ncols,const Double_t *data, Option_t option= "")`
- `TMatrixD(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb, const Double_t *data,Option_t *option="")`
- `TMatrixDSym(Int_t nrows)`
- `TMatrixDSym(Int_t row_lwb,Int_t row_upb)`
- `TMatrixDSym(Int_t nrows,const Double_t *data,Option_t *option="")`
- `TMatrixDSym(Int_t row_lwb,Int_t row_upb, const Double_t *data, Option_t *opt="")`
- `TMatrixDSparse(Int_t nrows,Int_t ncols)`
- `TMatrixDSparse(Int_t row_lwb,Int_t row_upb,Int_t col_lwb, Int_t col_upb)`
- `TMatrixDSparse(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb, Int_t nr_nonzeros,Int_t *row,Int_t *col,Double_t *data)`

Use one of the following methods to fill a matrix:

- `SetMatrixArray(const Double_t*data,Option_t*option="")`<br>
Copies array data. If `option="F"`, the array fills the matrix column-wise else row-wise. This option is implemented for `TMatrixD` and `TMatrixDSym`. It is expected that the array data contains at least `fNelems` entries.

- `SetMatrixArray(Int_t nr,Int_t *irow,Int_t *icol,Double_t *data)`<br>
Only available for sparse matrices. The three arrays should each contain `nr` entries with row index, column index and data entry. Only the entries with non-zero data value are inserted.

- `operator()`, `operator[]`<br>
These operators provide the easiest way to fill a matrix but are in particular for a sparse matrix expensive. If no entry for slot (`i`,`j`) is found in the sparse index table it will be entered, which involves some memory management. Therefore, before invoking this method in a loop set the index table first through a call to the `SetSparseIndex()` method.

- `SetSub(Int_t row_lwb,Int_t col_lwb,const TMatrixDBase &source)`<br>
The matrix to be inserted at position (`row_lwb`,`col_lwb`) can be both, dense or sparse.

- `Use()`<br>
Allows inserting another matrix or data array without actually copying the data.<br>
The following list shows the application of the `Use()` method.
- First for normal matrices:
   - `Use(TMatrixD &a)`
   - `Use(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Double_t *data)`
   - `Use(Int_t nrows,Int_t ncols,Double_t *data)`
- For symmetric matrices:
   - `Use(TMatrixDSym &a)`
   - `Use(Int_t nrows,Double_t *data)`
   - `Use(Int_t row_lwb,Int_t row_upb,Double_t *data)`
- For sparse matrices:
   - `Use(TMatrixDSparse &a)`
   - `Use(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Int_t nr_no nzeros, Int_t *pRowIndex,Int_t *pColIndex,Double_t *pData)`
   - `Use(Int_t nrows,Int_t ncols,Int_t nr_nonzeros,Int_t *pRowIndex,Int_t *pColIndex,Double_t *pData)`

_**Example**_

A Hilbert matrix is created by copying an array.

{% highlight C++ %}
   TMatrixD h(5,5);
   TArrayD data(25);
   for (Int_t = 0; i < 25; i++) {
      const Int_t ir = i/5;
      const Int_t ic = i%5;
      data[i] = 1./(ir+ic);
   }
   h.SetMatrixArray(data.GetArray());
{% endhighlight %}

You can also assign the data array to the matrix without actually copying it.

{% highlight C++ %}
   TMatrixD h; h.Use(5,5,data.GetArray());
   h.Invert();
{% endhighlight %}

The array data now contains the inverted matrix.

Now a unit matrix in sparse format is created.

{% highlight C++ %}
   TMatrixDSparse unit1(5,5);
   TArrayI row(5),col(5);
   for (Int_t i = 0; i < 5; i++) row[i] = col[i] = i;
   TArrayD data(5); data.Reset(1.);
   unit1.SetMatrixArray(5,row.GetArray(),col.GetArray(),data.GetArray());

   TMatrixDSparse unit2(5,5);
   unit2.SetSparseIndex(5);
   unit2.SetRowIndexArray(row.GetArray());
   unit2.SetColIndexArray(col.GetArray());
   unit2.SetMatrixArray(data.GetArray());
{% endhighlight %}

### Inverting a matrix

- Use the `Invert(Double_t &det=0)` function to invert a matrix:

{% highlight C++ %}
   TMatrixD a(...);
   a.Invert();
{% endhighlight %}

-- or --

- Use the appropriate constructor to invert a matrix:

{% highlight C++ %}
   TMatrixD b(kInvert,a);
{% endhighlight %}

Both methods are available for general and symmetric matrices.

For matrices whose size is less than or equal to 6x6, the `InvertFast(Double_t &det=0)` function is available. Here the Cramer algorithm will be applied, which is faster but less accurate.

#### Using decomposition classes for inverting

You can also use the following decomposition classes (see → [Matrix decompositions](#matrix-decompositions")) for inverting a matrix:

<table width="100%" border="0">
  <tbody>
    <tr>
      <th scope="col">Name</th>
      <th scope="col">Matrix type</th>
      <th scope="col">Comment</th>
    </tr>
    <tr>
      <td>TDecompLU</td>
      <td>General</td>
      <td></td>
    </tr>
    <tr>
      <td>TDecompQRH</td>
      <td>General</td>
      <td></td>
    </tr>
    <tr>
      <td>TDecompSVD</td>
      <td>General</td>
      <td>Can manipulate singular matrix.</td>
    </tr>
    <tr>
      <td>TDecompBK</td>
      <td>symmetric</td>
      <td></td>
    </tr>
    <tr>
      <td>TDecompChol</td>
      <td>Symmetric</td>
      <td>Matrix should also be positive definite.</td>
    </tr>
    <tr>
      <td>TDecompSparse</td>
      <td>Sparse</td>
      <td></td>
    </tr>
  </tbody>
</table>

If the required matrix type is general, you also can handle symmetric matrices.

_**Example**_

This example shows how to check whether the matrix is singular before attempting to invert it.

{% highlight C++ %}
   TDecompLU lu(a);
   TMatrixD b;
   if (!lu.Decompose()) {
      cout << "Decomposition failed, matrix singular ?" << endl;
      cout << "condition number = " << = a.GetCondition() << endl;
   } else {
      lu.Invert(b);
   }
{% endhighlight %}


### Matrix operators and methods

The matrix/vector operations are classified according to BLAS (basic linear algebra subroutines) levels.

#### Arithmetic operations between matrices

<table width="100%" border="0">
  <tbody>
    <tr>
      <th scope="col">Description</th>
      <th scope="col">Format</th>
      <th scope="col">Comment</th>
    </tr>
    <tr>
      <td>Element</td>
      <td>C=A+B</td>
      <td>overwrites A</td>
    </tr>
    <tr>
      <td>Wise sum</td>
      <td>A+=B<br>Add (A,alpha,B)
<br>TMatrixD(A,TMatrixD::kPlus,B)</td>
      <td>A = A + &alpha; B constructor</td>
    </tr>
        <tr>
      <td>Element wise substraction</td>
      <td>C=A-B A-=B<br>
TMatrixD(A,TMatrixD::kMinus,B)</td>
      <td>overwrites A<br>
constructor</td>
    </tr>
            <tr>
      <td>Matrix multiplication</td>
      <td>C=A*B<br>
A*=B<br>
C.Mult(A,B)<br>TMatrixD(A,TMatrixD::kMult,B)<br>TMatrixD(A, TMatrixD(A, TMatrixD::kTransposeMult,B)<br>TMatrixD(A, TMatrixD::kMultTranspose,B)</td>
      <td>overwrites A<br>&nbsp;<br>&nbsp;<br>constructor of A.B<br>constructor of A<sup>T</sup> .B<br>constructor of A.B<sup>T</sup></td>
    </tr>
      <tr>
      <td>Element wise multiplication</td>
      <td>ElementMult(A,B)</td>
      <td>A(i,j)*= B(i,j)</td>
    </tr>
      <tr>
      <td>Element wise division</td>
      <td>ElementDiv(A,B)</td>
      <td>A(i,j)/= B(i,j)</td>
    </tr>
  </tbody>
</table>

#### Arithmetic operations between matrices and real numbers

<table width="100%" border="0">
  <tbody>
    <tr>
      <th scope="col">Description</th>
      <th scope="col">Format</th>
      <th scope="col">Comment</th>
    </tr>
    <tr>
      <td>Element wise sum</td>
      <td>C=r+A C=A+r A+=r</td>
      <td>overwrites A</td>
    </tr>
    <tr>
      <td>Element wise subtraction</td>
      <td>C=r-A C=A-r A-=r</td>
      <td>overwrites A</td>
    </tr>
       <tr>
      <td>Matrix multiplication</td>
      <td>C=r*A C=A*r A*=r</td>
      <td>overwrites A</td>
    </tr>
  </tbody>
</table>

#### Comparison and Boolean operations

**Comparison between two matrices**

<table width="100%" border="0">
  <tbody>
    <tr>
      <th scope="col">Description</th>
      <th scope="col">Output</th>
      <th scope="col">Descriptions</th>
    </tr>
    <tr>
      <td>A == B</td>
      <td>Bool_t</td>
      <td>equal to</td>
    </tr>
    <tr>
      <td>A != B</td>
      <td>matrix</td>
      <td>not equal</td>
    </tr>
        <tr>
      <td>A > B</td>
      <td>matrix</td>
      <td>greater than</td>
    </tr>
        <tr>
      <td>A >= B</td>
      <td>matrix</td>
      <td>greater than or equal to</td>
    </tr>
        <tr>
      <td>A < B</td>
      <td>matrix</td>
      <td>smaller than</td>
    </tr>
        <tr>
      <td>A <= B</td>
      <td>matrix</td>
      <td>smaller than or equal to</td>
    </tr>
        <tr>
      <td>AreCompatible(A,B)</td>
      <td>Bool_t</td>
      <td>compare matrix properties</td>
    </tr>
        <tr>
      <td>Compare(A,B)</td>
      <td>Bool_t</td>
      <td>return summary of comparison</td>
    </tr>
        <tr>
      <td>VerifyMatrixIdentity(A,B,verb, maxDev)</td>
      <td>&nbsp;</td>
      <td>check matrix identity within maxDev tolerance</td>
    </tr>
  </tbody>
</table>

**Comparison between matrix and real number**

<table width="100%" border="0">
  <tbody>
    <tr>
      <th scope="col">Format</th>
      <th scope="col">Output</th>
      <th scope="col">Description</th>
    </tr>
    <tr>
      <td>A == r</td>
      <td>Bool_t</td>
      <td>equal to</td>
    </tr>
    <tr>
      <td>A != r</td>
      <td>Bool_t</td>
      <td>not equal</td>
    </tr>
        <tr>
      <td>A > r</td>
      <td>Bool_t</td>
      <td>greater than</td>
    </tr>
        <tr>
      <td>A >= r</td>
      <td>Bool_t</td>
      <td>greater than or equal to</td>
    </tr>
        <tr>
      <td>A < r</td>
      <td>Bool_t</td>
      <td>smaller than</td>
    </tr>
        <tr>
      <td>A <= r</td>
      <td>Bool_t</td>
      <td>smaller than or equal to</td>
    </tr>
        <tr>
      <td>VerifyMatrixValue(A,r,verb, maxDev)</td>
      <td>Bool_t</td>
      <td>compare matrix value with r within maxDev tolerance</td>
    </tr>
        <tr>
      <td>A.RowNorm()</td>
      <td>Double_t</td>
      <td>norm induced by the infinity vector norm</td>
    </tr>
        <tr>
      <td>A.NormInf()</td>
      <td>Double_t</td>
      <td>&nbsp;</td>
    </tr>
        <tr>
      <td>A.ColNorm()</td>
      <td>Double_t</td>
      <td>norm induced by the 1 vector norm</td>
    </tr>
        <tr>
      <td>A.Norm1()</td>
      <td>Double_t</td>
      <td>&nbsp;</td>
    </tr>
        <tr>
      <td>A.E2Norm()</td>
      <td>Double_t</td>
      <td>square of the Euclidean norm</td>
    </tr>
        <tr>
      <td>A.NonZeros()</td>
      <td>Int_t</td>
      <td>&nbsp;</td>
    </tr>
        <tr>
      <td>A.Sum()</td>
      <td>Double_t</td>
      <td>number of elements unequal zero</td>
    </tr>
        <tr>
      <td>A.Min()</td>
      <td>Double_t</td>
      <td>&nbsp;</td>
    </tr>
        <tr>
      <td>A.Max()</td>
      <td>Double_t</td>
      <td>&nbsp;</td>
    </tr>
        <tr>
      <td>A.NormByColumn (v,"D")</td>
      <td>TMatrixD</td>
      <td>&nbsp;</td>
    </tr>
        <tr>
      <td>A.NormByRow (v,"D")</td>
      <td>TMatrixD</td>
      <td>&nbsp;</td>
    </tr>
  </tbody>
</table>

### Matrix views

With the following matrix view classes, you can access the matrix elements:

- `TMatrixDRow`
- `TMatrixDColumn`
- `TMatrixDDiag`
- `TMatrixDSub`

#### Matrix view operators

For the matrix view classes `TMatrixDRow`, `TMatrixDColumn` and `TMatrixDDiag`, the necessary assignment operators are available to interact with the vector class `TVectorD`.<br>The sub matrix view classes `TMatrixDSub` has links to the matrix classes `TMatrixD` and `TMatrixDSym.`

The next table summarizes how to access the individual matrix elements in the matrix view classes.

<table width="100%" border="0">
  <tbody>
    <tr>

      <th scope="col">Format</th>
      <th scope="col">Comment</th>
    </tr>
    <tr>

      <td>TMatrixDRow(A,i)(j) TMatrixDRow(A,i)[j]</td>
      <td>element A<sub>ij</sub></td>
    </tr>
    <tr>

      <td>TMatrixDColumn(A,j)(i) TMatrixDColumn(A,j)[i]</td>
      <td>element A<sub>ij</sub></td>
    </tr>
        <tr>

      <td>TMatrixDDiag(A(i) TMatrixDDiag(A[i]</td>
      <td>element A<sub>ij</sub></td>
    </tr>
    <tr>

      <td>TMatrixDSub(A(i) TMatrixDSub(A,rl,rh,cl,ch)(i,j)</td>
      <td>element A<sub>ij</sub><br>element A<sub>rl+i,cl+j</sub></td>
    </tr>
  </tbody>
</table>

#### Matrix decompositions

There are the following classes available for matrix decompositions:

- {% include ref class="TDecompLU" %}: Decomposes a general `n x n` matrix `A` into `P A = L U`.
- {% include ref class="TDecompBK" %}: The Bunch-Kaufman diagonal pivoting method decomposes a real symmetric matrix `A`.
- {% include ref class="TDecompChol" %}: The Cholesky decomposition class, which decomposes a symmetric, positive definite matrix `A = U^T * U` where `U` is a upper triangular matrix.
- {% include ref class="TDecompQRH" %}: QR decomposition class.
- {% include ref class="TDecompSVD" %}: Single value decomposition class.
- {% include ref class="TDecompSparse" %}: Sparse symmetric decomposition class.

### Matrix Eigen analysis

With the `TMatrixDEigen` and `TMatrixDSymEigen` classes, you can compute eigenvalues and eigenvectors for general dense and symmetric real matrices.


## Additonal Notes


 The present package provides all facilities to completely AVOID
 returning matrices. Use "TMatrixD A(TMatrixD::kTransposed,B);"
 and other fancy constructors as much as possible. If one really needs
 to return a matrix, return a TMatrixTLazy object instead. The
 conversion is completely transparent to the end user, e.g.
 "TMatrixT m = THaarMatrixT(5);" and _is_ efficient.

 Since TMatrixT et al. are fully integrated in ROOT, they of course
 can be stored in a ROOT database.

### How to efficiently use this package

#### 1. Never return complex objects (matrices or vectors)
   Danger: For example, when the following snippet:
~~~
   TMatrixD foo(int n)
   {
     TMatrixD foom(n,n); fill_in(foom); return foom;
   }
   TMatrixD m = foo(5);
~~~
   runs, it constructs matrix foo:foom, copies it onto stack as a
   return value and destroys foo:foom. Return value (a matrix)
   from foo() is then copied over to m (via a copy constructor),
   and the return value is destroyed. So, the matrix constructor is
   called 3 times and the destructor 2 times. For big matrices,
   the cost of multiple constructing/copying/destroying of objects
   may be very large. *Some* optimized compilers can cut down on 1
   copying/destroying, but still it leaves at least two calls to
   the constructor. Note, TMatrixDLazy (see below) can construct
   TMatrixD m "inplace", with only a _single_ call to the
   constructor.

#### 2. Use "two-address instructions"
~~~
   "void TMatrixD::operator += (const TMatrixD &B);"
~~~
    as much as possible.
    That is, to add two matrices, it's much more efficient to write
~~~
   A += B;
~~~
    than
~~~
   TMatrixD C = A + B;
~~~
   (if both operand should be preserved,
     TMatrixD C = A; C += B;
   is still better).

#### 3. Use glorified constructors when returning of an object seems inevitable:
~~~
   "TMatrixD A(TMatrixD::kTransposed,B);"
   "TMatrixD C(A,TMatrixD::kTransposeMult,B);"
~~~

  like in the following snippet (from $ROOTSYS/test/vmatrix.cxx)
  that verifies that for an orthogonal matrix T, T'T = TT' = E.

~~~
   TMatrixD haar = THaarMatrixD(5);
   TMatrixD unit(TMatrixD::kUnit,haar);
   TMatrixD haar_t(TMatrixD::kTransposed,haar);
   TMatrixD hth(haar,TMatrixD::kTransposeMult,haar);
   TMatrixD hht(haar,TMatrixD::kMult,haar_t);
   TMatrixD hht1 = haar; hht1 *= haar_t;
   VerifyMatrixIdentity(unit,hth);
   VerifyMatrixIdentity(unit,hht);
   VerifyMatrixIdentity(unit,hht1);
~~~

#### 4. Accessing row/col/diagonal of a matrix without much fuss
   (and without moving a lot of stuff around):

~~~
   TMatrixD m(n,n); TVectorD v(n); TMatrixDDiag(m) += 4;
   v = TMatrixDRow(m,0);
   TMatrixDColumn m1(m,1); m1(2) = 3; // the same as m(2,1)=3;
~~~
   Note, constructing of, say, TMatrixDDiag does *not* involve any
   copying of any elements of the source matrix.

#### 5. It's possible (and encouraged) to use "nested" functions
   For example, creating of a Hilbert matrix can be done as follows:

~~~
   void foo(const TMatrixD &m)
   {
    TMatrixD m1(TMatrixD::kZero,m);
    struct MakeHilbert : public TElementPosActionD {
       void Operation(Double_t &element)
          { element = 1./(fI+fJ-1); }
    };
    m1.Apply(MakeHilbert());
   }
~~~

   of course, using a special method THilbertMatrixD() is
   still more optimal, but not by a whole lot. And that's right,
   class MakeHilbert is declared *within* a function and local to
   that function. It means one can define another MakeHilbert class
   (within another function or outside of any function, that is, in
   the global scope), and it still will be OK. Note, this currently
   is not yet supported by the interpreter CINT.

   Another example is applying of a simple function to each matrix element:

~~~
   void foo(TMatrixD &m,TMatrixD &m1)
   {
    typedef  double (*dfunc_t)(double);
    class ApplyFunction : public TElementActionD {
       dfunc_t fFunc;
       void Operation(Double_t &element)
            { element=fFunc(element); }
     public:
       ApplyFunction(dfunc_t func):fFunc(func) {}
    };
    ApplyFunction x(TMath::Sin);
    m.Apply(x);
   }
~~~

   Validation code $ROOTSYS/test/vmatrix.cxx and vvector.cxx contain
   a few more examples of that kind.

#### 6. Lazy matrices:
   instead of returning an object return a "recipe"
   how to make it. The full matrix would be rolled out only when
   and where it's needed:
~~~
   TMatrixD haar = THaarMatrixD(5);
~~~
   THaarMatrixD() is a *class*, not a simple function. However
   similar this looks to a returning of an object (see note #1
   above), it's dramatically different. THaarMatrixD() constructs a
   TMatrixDLazy, an object of just a few bytes long. A special
   "TMatrixD(const TMatrixDLazy &recipe)" constructor follows the
   recipe and makes the matrix haar() right in place. No matrix
   element is moved whatsoever!

### Acknowledgements

 1. Oleg E. Kiselyov
    First implementations were based on the his code . We have diverged
    quite a bit since then but the ideas/code for lazy matrix and
    "nested function" are 100% his .
    You can see him and his code in action at http://okmij.org/ftp
 2. Chris R. Birchenhall,
    We adapted his idea of the implementation for the decomposition
    classes instead of our messy installation of matrix inversion
    His installation of matrix condition number, using an iterative
    scheme using the Hage algorithm is worth looking at !
    Chris has a nice writeup (matdoc.ps) on his matrix classes at
    ftp://ftp.mcc.ac.uk/pub/matclass/
 3. Mark Fischler and Steven Haywood of CLHEP
    They did the slave labor of spelling out all sub-determinants
    for Cramer inversion  of (4x4),(5x5) and (6x6) matrices
    The stack storage for small matrices was also taken from them
 4. Roldan Pozo of TNT (http://math.nist.gov/tnt/)
    He converted the EISPACK routines for the eigen-vector analysis to
    C++ . We started with his implementation
 5. Siegmund Brandt (http://siux00.physik.uni-siegen.de/~brandt/datan
    We adapted his (very-well) documented SVD routines


*/
