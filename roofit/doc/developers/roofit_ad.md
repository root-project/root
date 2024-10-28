\defgroup roofit_dev_docs_ad How to extend the use of Automatic Differentiation in RooFit
\ingroup roofit_dev_docs
\date October 2023
\brief Developer guide on how to add support for Automatic Differentiation via code generation.

#  How to extend the use of Automatic Differentiation in RooFit

## What is RooFit?

[RooFit] is a statistical data analysis tool, widely used in scientific
research, especially in the high-energy physics (HEP) field. It is an
extension of the ROOT framework, a C++ based data analysis framework that
provides tools for data storage, analysis, and visualization. RooFit provides
a set of tools/classes to define and evaluate probability density functions
(PDFs), perform maximum likelihood fits, perform statistical tests, etc.


## Proof of Concept: Speeding up RooFit using Automatic Differentiation (AD)

RooFit is used to reduce statistical models (functions) to find a set of
parameters that minimize the value of the function. This minimization happens
via one of several methods relying heavily on the computation of derivatives
of the function with respect to its free parameters. Currently, the
computation of Numerical Derivatives is the most time-consuming component of
RooFit [^1]. On the other hand, derivatives computed using the Automatic
Differentiation tool [Clad] have been shown to be far more efficient [^2].

\htmlonly
<div class="pyrootbox">
\endhtmlonly

Main Advantage of using AD with RooFit: efficient and more precise
derivatives. It computes derivatives with high precision, avoiding the errors
that may arise from approximating derivatives using finite differences.

\htmlonly
</div>
\endhtmlonly

### AD Support essentially requires Code Generation

As we'll discuss in upcoming sections, *AD support* can be added using *C++
Code generation*.
These two terms may be used interchangeably in this document, since the term
*Code Generation* better helps visualize the transformation that is enabling
AD support.

## Current Status of Code Generation in RooFit

RooFit is an extensive toolkit.
The initiative to add AD support/ Code Generation has been started, but has
not yet achieved full coverage for the models defined/maintained in RooFit.

## How Clad enables AD support using Source Code Transformation

[Clad] is a C++ plugin for Clang. It implements a technique called Source Code
 Transformation to enable AD support.

Source Code Transformation takes the source code (that needs to be
differentiated) as the input and generates an output code that represents the
derivative of the input. This output code can be used instead of the input
code for more efficient compilation.

For more technical details, please see the following paper:

> [Automatic Differentiation of Binned Likelihoods with RooFit and Clad](https://arxiv.org/abs/2304.02650)

## Overview on RooFit implementation details to access source code transformation AD

In RooFit jargon, what is meant by a "RooFit class" is a class inheriting from
RooAbsArg that represents a mathematical function, a PDF, or any other
transformation of inputs that are also represented by RooAbsArg objects.
Almost all final classes deriving from RooAbsArg have RooAbsReal as an
intermediate base class, which is the base class for all RooAbsArg that
represent real-valued nodes in the computation graph.
As such RooFit objects are so prevalent in practice, the names RooAbsArg and
RooAbsReal are used interchangeably in this guide.

Users take these classes to build a computational graph that represents the
PDF (also called "model") that they want to use for fitting the data.
The user then passes his final PDF and a RooAbsData object to the
RooAbsPdf::fitTo() method, which implicitly creates a negative-log likelihood
(NLL) that RooFit minimizes for parameter estimation.
The NLL object, internally created by RooAbsPdf::createNLL(), is a RooAbsArg
itself.
In technical terms, it's another larger computation graph that encompasses the
 computation graph representing the PDF.

To enable source code transformation AD for RooFit NLLs with Clad, RooFit has a
routine that can traverse a computation graph made of RooAbsArg objects and
transform it to much simpler C++ code that mathematically represents the same
computation, but without any overhead that is hard to digest by the AD tool.

On a high level, this *code generation* is implemented as follows:

1. The computation graph is visited recursively by a
   RooFit::CodegenContext object, via the
   RooFit::codegenImpl(RooAbsArg &, RooFit::CodegenContext &) function that implements the translation of a
   given RooFit class to minimal C++ code. This is an example of the visitor
   pattern.

2. The generated code is processed by a RooFuncWrapper object, which takes care
   of just-in-time compiling it with the ROOT interpreter, generating the
   gradient code with Clad, and compiling that as well.

3. Since the RooFuncWrapper is implementing a RooAbsArg itself, it can now be
   used as a drop-in replacement for the RooAbsArg that was the top node of the
   original computation graph, with the added benefit that it can be queried for
   the gradient.

In summary, the important ingredient to enable AD in RooFit is to support the
**C++ code generation** from RooFit classes.

# Steps to enable Code Generation in RooFit classes

There are multiple code generation approaches that can be followed to add Code
 Generation support in RooFit classes.

**Approach 1:** For very simple cases like `RooRatio`, it may be preferable to
 write the entire code in a single string.

**Approach 2:** Another approach could be to extract free functions in a
separate header file.
Since Clad prefers the code for models to be within a single translation unit,
 in many classes, this has been implemented by moving the computational
aspects of the RooFit class
to free functions in a single header file named [MathFuncs].
This approach enables easier debugging
(e.g., you can standalone-compile the generated code with just a few header
files copied outside ROOT).

*Refactoring* It is important to refactor the code such that:

- the footprint of the generated code is minimized by referring to existing
functions with the definition known by interpreter (i.e., they are in public
header files).

- to reuse common code, both in the generated code, and in the existing
RooAbsReal::evaluate() method (meaning that the refactoring of `evaluate()` is
optional, but it is recommended).

\htmlonly
<div class="pyrootbox">
\endhtmlonly

*Implement Code Generation support in custom classes*: Framework developers
that want to implement Code Generation support for their custom classes, this
approach of extracting free functions in a separate header file is not
suitable, since they can't put the code in a header that is part of the ROOT
installation. Please note the following recommendations:

- while developing your custom class, add these functions to your classes
header file (e.g., as part of the class definition), and

- if/when your class is upstreamed to RooFit, expect to move into the
`RooFit::Detail` namespace and their implementations into `MathFuncs.h`.

\htmlonly
</div>
\endhtmlonly

*Overloading the code generation function*: The `RooFit::codegenImpl()` function
needs to be overloaded to specify how the class is translating to C++ code
that is using the aforementioned free function.

**Sample Steps**: To add Code Generation support to an existing RooFit class,
following is a sample set of steps (using the aforementioned approach of
extracting free functions in a separate file.).

**1. Extract logic into a separate file** Implement what your class is
supposed to do as a free function in [MathFuncs].
This implementation must be compatible with the syntax supported by Clad.

**2. Refactor evaluate():** Refactor the existing `RooAbsReal::evaluate()`
 function to use the `MathFuncs.h` implementation. This is optional, but
can reduce code duplication and potential for bugs. This may require some
effort if an extensive caching infrastructure is used in your model.

**3. Add RooFit::codegenImpl():** Define a (typically) simple
 `RooFit::codegenImpl()` function that extracts the mathematically differentiable
properties out of the RooFit classes that make up the statistical model.
This function needs to be declared in a public header file, such that it is known to the interpreter.

The `RooFit::codegenImpl()` function helps implement the code generation logic that is
used to optimize numerical evaluations. It accomplishes this by using a small
subset of helper functions that are available in the
`RooFit::CodegenContext` class (see Appendix B).
It converts a RooFit expression into a form that can be
efficiently evaluated by Clad.

The `RooFit::codegenImpl()` function places a `std::string` inside the `RooFit::CodegenContext`.
This string is representing the
underlying mathematical notation of the class as code, that can later be
concatenated into a single string representing the entire model. This string
of code is then just-in-time compiled by Cling (a C++ interpreter for Root).

**4. RooFit::codegenIntegralImpl() Use Case:** If your class includes (or should
include) the `analyticalIntegral()` function, then a simple
`RooFit::codegenIntegralImpl()` function needs to be defined to help call the
`analyticalIntegral()` function.


# Example for adding Code Generation support to RooFit classes

Let us take the `RooPoisson.cxx` class as an example.

> [roofit/roofit/src/RooPoisson.cxx](https://github.com/root-project/root/blob/master/roofit/roofit/src/RooPoisson.cxx)

First step is to locate the `RooPoisson::evaluate()` function. Most RooFit
classes implement this function.

> RooFit internally calls the `evaluate()` function to evaluate a single node
 in a compute graph.

## Before Code Generation Support

Following is a code snippet from `RooPoisson` *before* it had AD support.

``` {.cpp}
double RooPoisson::evaluate() const
{
  double k = _noRounding ? x : floor(x);
  if(_protectNegative && mean<0) {
    RooNaNPacker np;
    np.setPayload(-mean);
    return np._payload;
  }
  return TMath::Poisson(k,mean);
}
```
`TMath::Poisson()` is a simple mathematical function. For this example, the
relevant part is `return TMath::Poisson(k,mean);`. This needs to be extracted
into the `MathFuncs.h` file and the fully qualified name of the function
referencing that file should be used here instead.

## After Code Generation Support

Following is a code snippet from `RooPoisson` *after* it has AD support.

### Step 1. Refactor the `RooPoisson::evaluate()` Function

``` {.cpp}
/// Implementation in terms of the TMath::Poisson() function.

double RooPoisson::evaluate() const
{
  double k = _noRounding ? x : floor(x);
  if(_protectNegative && mean<0) {
    RooNaNPacker np;
    np.setPayload(-mean);
    return np._payload;
  }
  return RooFit::Detail::MathFuncs::poisson(k, mean);
}
```

Note that the `evaluate()` function was refactored in such a way that the
mathematical parts were moved to an inline function in a separate header file
named `MathFuncs`, so that Clad could see and differentiate that function.
The rest of the contents of the function remain unchanged.

> All contents of the `evaluate()` function don't always need to be pulled
out, only the required parts (mathematical  logic) should be moved to
`MathFuncs`.

**What is MathFuncs?**

Moving away from the class-based hierarchy design, `MathFuncs.h` a simply
a flat file of function implementations.

This file is required since Clad will not be able to see anything that is not
inlined and explicitly available to it during compilation (since it has to be
in the same translation). So other than of generating these functions on the
fly, your only other option is to place these functions in a separate header
file and make them inline.

Theoretically, multiple header files can also be used and then mashed
together.

> Directory path: [roofit/roofitcore/inc/RooFit/Detail/MathFuncs.h](https://github.com/root-project/root/blob/master/roofit/roofitcore/inc/RooFit/Detail/MathFuncs.h)

### Step 2. Overload RooFit::codegenImpl()

**RooFit::codegenImpl() Example 1:** Continuing our RooPoisson example:

To translate the `RooPoisson` class, create a code generation function and in it
include a call to the free function.

``` {.cpp}
void RooFit::codegenImpl(RooPoisson &arg, RooFit::CodegenContext &ctx)
{
   std::string xName = ctx.getResult(arg.getX());
   if (!arg.getNoRounding())
      xName = "std::floor(" + xName + ")";

   ctx.addResult(&arg, ctx.buildCall("RooFit::Detail::MathFuncs::poisson", xName, arg.getMean()));
}
```

Here we can see that the name of the variable `x` (remember that "x" is a
member of RooPoisson) is retrieved and stored in the `xName` variable. Next,
there's an `if` condition that does an operation on `x` (may or may not round
it to the nearest integer, depending on the condition).

The important part is where the `RooFit::CodegenContext::addResult()` function adds
 the result of evaluating the Poisson function to the context (`ctx`). It uses
the `RooFit::CodegenContext::buildCall()` method to construct a function call to the fully
qualified name of `MathFuncs::poissonEvaluate` (which now resides in the
`MathFuncs` file), with arguments `xName` and the mean of the Poisson.

Essentially, the `RooFit::codegenImpl()` function constructs a function call
 to evaluate the Poisson function using 'x' and 'mean' variables, and adds the
result to the context.

Helper Functions:

- `RooFit::CodegenContext::getResult()` helps lookup the result of a child node (the string that the
child node previously saved in a variable using the `RooFit::CodegenContext::addResult()` function).

- `RooFit::CodegenContext::addResult()` It may include a function call, an expression, or something
more complicated. For a specific class, it will add whatever is represented on
 the right-hand side to the result of that class, which can then be propagated
 in the rest of the compute graph.

\note For each `RooFit::codegenImpl()` function, it is important to call `RooFit::CodegenContext::addResult()` since this is what enables the squashing to happen.


**translate() Example 2:** Following is a code snippet from `RooGaussian.cxx`
*after* it has AD support.

``` {.cpp}
void RooFit::codegenImpl(RooGaussian &arg, RooFit::CodegenContext &ctx)
{
   // Build a call to the stateless gaussian defined later.
   ctx.addResult(&arg, ctx.buildCall("RooFit::Detail::MathFuncs::gaussian", arg.getX(), arg.getMean(), arg.getSigma()));
}
```

Here we can see that the `RooFit::codegenImpl(RooGaussian &, RooFit::CodegenContext &)` function constructs a
function call using the `RooFit::CodegenContext::buildCall()` method. It specifies the fully qualified
name of the `gaussian` function (which is now part of the
`MathFuncs` file), and includes the x, mean, and sigma variables as
arguments to this function call.

Helper Function:

- `RooFit::CodegenContext::buildCall()` helps build a function call. Requires the fully qualified name
 (`RooFit::Detail::MathFuncs::gaussian`) of the function. When
this external `RooFit::CodegenContext::buildCall()` function is called, internally, the `RooFit::CodegenContext::getResult()`
function is called on the input RooFit objects (e.g., x, mean, sigma). That's
the only way to propagate these upwards into the compute graph.

**translate() Example 3:** A more complicated example of a `RooFit::codegenImpl()`
function can be seen here:

``` {.cpp}
void RooFit::codegenImpl(RooFit::Detail::RooNLLVarNew &arg, RooFit::CodegenContext &ctx)
{
   std::string weightSumName = ctx.makeValidVarName(GetName()) + "WeightSum";
   std::string resName = ctx.makeValidVarName(GetName()) + "Result";
   ctx.addResult(this, resName);
   ctx.addToGlobalScope("double " + weightSumName + " = 0.0;\n");
   ctx.addToGlobalScope("double " + resName + " = 0.0;\n");

   const bool needWeightSum = _expectedEvents || _simCount > 1;

   if (needWeightSum) {
      auto scope = ctx.beginLoop(this);
      ctx.addToCodeBody(weightSumName + " += " + ctx.getResult(*_weightVar) + ";\n");
   }

   if (_simCount > 1) {
      std::string simCountStr = std::to_string(static_cast<double>(_simCount));
      ctx.addToCodeBody(resName + " += " + weightSumName + " * std::log(" + simCountStr + ");\n");
   }
 ...

}
```

> Source: - [RooNLLVarNew](https://github.com/root-project/root/blob/master/roofit/roofitcore/src/RooNLLVarNew.cxx)

The complexity of the `RooFit::codegenImpl()` function in this example can
 be attributed to the more complex scenarios/operations specific to the
computation of negative log-likelihood (NLL) values for probability density
functions (PDFs) in RooFit, especially for simultaneous fits (multiple
simultaneous PDFs being considered) and binned likelihoods (adding further
complexity).

In this example, the `RooFit::codegenImpl()` function generates code to
compute the Negative Log likelihood (NLL). We can see that the intermediate
result variable `resName` is added to the context so that it can be accessed
 and used in the generated code. This variable is made available globally
(using `addToGlobalScope()`).

If a weight sum is needed, then it creates a loop, and `weightSumName` is
accumulated with the weight variable. Otherwise, if there are multiple
simultaneous PDFs, then it adds a term to the result that scales with the
logarithm of the count of simultaneous PDFs. The rest of the function body
(including the loop scope with NLL computation) has omitted from this example
to keep it brief.

Helper functions:

- `makeValidVarName()` helps get a valid name from the name of the respective
RooFit class. It then helps save it to the variable that represents the result
 of this class (the squashed code/ C++ function that will be created).

- `addToGlobalScope()` helps declare and initialize the results variable, so
that it can be available globally (throughout the function body). For local
variables, the `addToCodeBody()` function can be used to keep the variables in
 the respective scope (for example, within a loop).

- `beginLoop()` helps build the start and the end of a For loop for your
class. Simply place this function in the scope and place the contents of the
`For` loop below this statement. The code squashing task will automatically
build a loop around the statements that follow it. There's no need to worry
about the index of these loops, because they get propagated. For example, if
you want to iterate over a vector of RooFit objects using a loop, you don't
have to think about indexing them properly because the `beginLoop()` function
takes care of that. Simply call this function, place your function call in a
scope and after the scope ends, the loop will also end.

- `addToCodeBody()` helps add things to the body of the C++ function that
you're creating. It takes whatever string is computed in its arguments and
adds it to the overall function string (which will later be just-in-time
compiled). The `addToCodeBody()` function is important since not everything
can be added in-line and this function helps split the code into multiple
lines.


### Step 3. analyticalIntegral() Use Case

> Besides the `evaluate()` function, this tutorial illustrates how the
`analyticalIntegral()` can be updated. This highly dependent on the class that
 is being transformed for AD support, but will be necessary in those specific
instances.

Let's consider a fictional class RooFoo, that performs some arbitrary
mathematical operations called 'Foo' (as seen in doFoo() function below).

> Note that doFoo is a simplified example, in many cases the mathematical
operations are not limited to a single function, so they need to be spotted
within the `evaluate()` function.

``` {.cpp}
class RooFoo : public RooAbsReal {
    int a;
    int b;
    int doFoo() { return a* b + a + b; }
    int integralFoo() { return /* whatever */;}
    public:
    // Other functions...
    double evaluate() override {
        // Do some bookkeeping
        return doFoo();
    };
    double analyticalIntegral(Int_t code, const char* rangeName) override {
        // Select the right paths for integration using codes or whatever.
        return integralFoo();
    }
};
```

\note All RooFit classes are deriving from the RooAbsReal object, but
its details are not relevant to the current example.

Note how the `evaluate()` function overrides the `RooAbsReal` for the RooFoo
class. Similarly, the `analyticalIntegral()` function has also been overridden
 from the `RooAbsReal` class.

The `evaluate()` function includes some bookkeeping steps (commented out in
above example) that are not relevant to AD. The important part is that it
calls a specific function (doFoo() in this example), and returns the results.

Similarly, the `analyticalIntegral()` function calls a specific function (
`integralFoo()` in this example), and returns the results. It may also include
 some code that may need to be looked at, but for simplicity, its contents are
 commented out in this example.

#### Adding Code Generation Support to RooFoo class

Before creating the translate() function, move the mathematical logic (
`doFoo()` function in this example) out of the source class (RooFoo in this
example) and into a separate header file called `MathFuncs.h`. Also note
that the parameters a and b have been defined as inputs, instead of them just
being class members.

``` {.cpp}
///// The MathFuncs.h file
int doFoo(int a, int b) { return a* b + a + b; }
```

> Directory path: [roofit/roofitcore/inc/RooFit/Detail/MathFuncs.h](https://github.com/root-project/root/blob/master/roofit/roofitcore/inc/RooFit/Detail/MathFuncs.h)

So now that the `doFoo()` function exists in the `MathFuncs` namespace, we
 need to comment out its original function definition in the RooFoo class and
also add the namespace `MathFuncs` to wherever `doFoo()` it is referenced
(and also define input parameters for it).

``` {.cpp}
class RooFoo : public RooAbsReal {
    ...
    // int doFoo() { return a* b + a + b; }

    double evaluate() override {
        ...
        return MathFuncs::doFoo(a, b);
    };
 ```

Next, create the translate function. Most translate functions include a
`buildCall()` function, that includes the fully qualified name (including
'MathFuncs') of the function to be called along with the input parameters
as they appear in the function (a,b in the following example).

Also, each `translate()` function requires the `addResult()` function. It will
 add whatever is represented on the right-hand side to the result (saved in
the `res` variable in the following example) of this class, which can then be
propagated in the rest of the compute graph.

``` {.cpp}
     void translate(RooFit::CodegenContext &ctx) const override {
            std::string res = ctx.buildCall("MathFuncs::doFoo", a, b);
            ctx.addResult(this, res);
    }

```

#### When to add the buildCallToAnalyticIntegral() function

Besides creating the `translate()` function, the
`buildCallToAnalyticIntegral()` function also needs to be added when
`analyticalIntegral()` is found in your class. Depending on the code, you can
call one or more integral functions using the `code` parameter. Our RooFoo
example above only contains one integral function (`integralFoo()`).

Similar to `doFoo()`, comment out `integralFoo()' in the original file and
move it to 'MathFuncs.h'.

As with `doFoo()`. add the relevant inputs (a,b) as parameters, instead of
just class members.

``` {.cpp}
///// The MathFuncs.h file
int integralFoo(int a, int b) { return /* whatever */;}
```

> Directory path: [hist/hist/src/MathFuncs.h](https://github.com/root-project/root/blob/master/hist/hist/src/MathFuncs.h)

Next, in the original RooFoo class, update all references to the
`integralFoo()` function with its new fully qualified path (
`EvaluateFunc::integralFoo`) and include the input parameters as well (
`EvaluateFunc::integralFoo(a, b)`).

``` {.cpp}
    double analyticalIntegral(Int_t code, const char* rangeName) override {
        // Select the right paths for integration using codes or whatever.
        return EvaluateFunc::integralFoo(a, b);
    }
```

Next, in the `RooAbsReal::buildCallToAnalyticIntegral()` function, simply
return the output using the `buildCall()` function.

``` {.cpp}
    std::string
    buildCallToAnalyticIntegral(Int_t code, const char *rangeName, RooFit::CodegenContext &ctx) const override {
        return ctx.buildCall("EvaluateFunc::integralFoo", a, b);
    }
```

\note The implementation of the `RooAbsReal::buildCallToAnalyticIntegral()`
function is quite similar to the `translate()` function, except that in
`translate()`, you have to add to the result (using `addResult()`), while for
`buildCallToAnalyticIntegral()`, you only have to return the string (using
`buildCall()`).

**Consolidated Code changes in RooFoo example**

Final RooFoo code:

``` {.cpp}
class RooFoo : public RooAbsReal {
    int a;
    int b;
    // int doFoo() { return a* b + a + b; }
    // int integralFoo() { return /* whatever */;}
    public:
    // Other functions...
    double evaluate() override {
        // Do some bookkeeping
        return EvaluateFunc::doFoo(a, b);
    };
    double analyticalIntegral(Int_t code, const char* rangeName) override {
        // Select the right paths for integration using codes or whatever.
        return EvaluateFunc::integralFoo(a, b);
    }

    //// ************************** functions for AD Support ***********************
    void translate(RooFit::CodegenContext &ctx) const override {
        std::string res = ctx.buildCall("EvaluateFunc::doFoo", a, b);
        ctx.addResult(this, res);
    }

    std::string
    buildCallToAnalyticIntegral(Int_t code, const char *rangeName, RooFit::CodegenContext &ctx) const override {
        return ctx.buildCall("EvaluateFunc::integralFoo", a, b);
    }
    //// ************************** functions for AD Support ***********************
};

```

Mathematical code moved to `MathFuncs.h` file.

``` {.cpp}
int doFoo(int a, int b) { return a* b + a + b; }
```

Integrals moved to the 'MathFuncs.h' file.

``` {.cpp}
int integralFoo(int a, int b) { return /* whatever */;}
```

> Remember, as long as your code is supported by Clad (e.g., meaning there are
 custom derivatives defined for all external Math library functions used in
your code), it should work for AD support efforts. Please view Clad
documentation for more details.

---

## Appendix A - What could go wrong (FAQs)

### Will my analyticalIntegral() function support AD?

Both scenarios are possible:

1 - where `analyticalIntegral()` will be able to support AD

2 - where `analyticalIntegral()` will *not* be able to support AD

This requires further research.

### What if my evaluate() function cannot support AD?

In some cases. the `evaluate()` function is written in a piece-wise format
(multiple evaluations based on multiple chunks of code). You can review the
`MathFuncs.h` file to find AD support for several piece-wise (`if code==1
{...} else if code==2 {...}` ) code snippets.

However, there may still be some cases where AD support may not be possible
due to the way that `evaluate()` function works in that instance.

### What if my evaluate() function depends heavily on caching?

For simple caching, the caching logic can be separated from the
mathematical code that is being moved to `MathFuncs.h`, so that it can
retained in the original file.

For more complicated scenarios, the `code` variable can be used to identify
use cases (parts of the mathematical code in `evaluate()`) that should be
supported, while other parts that are explicitly not be supported (e.g., using
 `if code==1 {...} else if code==2 {...}`).

### Can classes using Numerical Integration support AD?

So far, no. This needs further exploration. Hint: classes using Numerical
Integration can be identified with the absence of the `analyticalIntegral()`
function.

### Why is my code falling back to Numeric Differentiation?

If you call in to an external Math library, and you use a function that has a
customized variant with an already defined custom derivative, then you may see
 a warning like "falling back to Numeric Differentiation". In most such cases,
 your derivative should still work, since Numeric Differentiation is already
well-tested in Clad.

To handle this, either define a custom derivative for that external function,
or find a way to expose it to Clad.

An example of this can be seen with `gamma_cdf()` in MathFuncs.h`,
for which the custom derivative is not supported, but in this specific
instance, it falls back to Numeric Differentiation and works fine, since `
gamma_cdf()` doesn't have a lot of parameters.

> In such cases, Numeric Differentiation fallback is only used for that
specific function. In above example, `gamma_cdf()` falls back to Numeric
Differentiation but other functions in `MathFuncs.h` will still be
able to use AD. This is because Clad is going to assume that you have a
derivative for this `gamma_cdf()` function, and the remaining functions will
use AD as expected. In the end, the remaining functions (including
`gamma_cdf()`) will try to fall back to Numeric Differentiation.

However, if you want to add pure AD support, you need to make sure that all
your external functions are supported by Clad (meaning there is a custom
derivative defined for each of them).

### How do I test my new class while adding AD support?

Please look at the test classes that test the derivatives, evaluates,
fixtures, etc. (defined in 'roofit/roofitcore/test'). You can clone and adapt
these tests to your class as needed. For example:

> [roofit/roofitcore/test/testRooFuncWrapper.cxx](https://github.com/root-project/root/blob/master/roofit/roofitcore/test/testRooFuncWrapper.cxx)

> Tip: Tests like above can be referenced to see which parts of RooFit already
 support AD.

### How do I control my compile time?

This is an area of research that still needs some work. In most cases, the
compile times are reasonable, but with an increase in the level of complexity,
 higher compile times may be encountered.


## Appendix B - Where does AD Logic Implementation reside?

Following classes provide several Helper Functions to translate existing logic
into AD-supported logic.

a - RooFit::CodegenContext

b - RooFuncWrapper

### a. RooFit::CodegenContext

> [roofit/roofitcore/inc/RooFit/CodegenContext.h](https://github.com/root-project/root/blob/master/roofit/roofitcore/inc/RooFit/CodegenContext.h)

It handles how to create a C++ function out of the compute graph (which is
created with different RooFit classes). This C++ function will be independent
of these RooFit classes.

RooFit::CodegenContext helps traverse the compute graph received from RooFit and
then it translates that into a single piece of code (a C++ function), that can
then be differentiated using Clad. It also helps evaluate the model.

In RooFit, evaluation is done using the 'evaluate()' function. It also
performs a lot of book-keeping, caching, etc. that is required for RooFit (but
not necessarily for AD).

A new `translate()` function is added to RooFit classes that includes a call
to this `evaluate()` function. `translate()` helps implement the Code
Squashing logic. All RooFit classes that should support AD need to use this
function. It creates a string of code, which is then just-in-time compiled
using Cling (C++ interpreter for ROOT). For each of the `translate()`
functions, it is important to call `addResult()` since this is what enables
the squashing to happen.

#### Helper Functions

- **RooFit::CodegenContext**: this class maintains the context for squashing of
RooFit models into code.  It keeps track of the results of various
expressions to avoid redundant calculations.

- **Loop Scopes()**: `beginloop()` and `endloop()` are used to create a scope
for iterating over vector observables (collections of data). This is
especially useful when dealing with data that comes in sets or arrays.

- **addToGlobalScope()**: helps add code statements to the global scope
(e.g., to declare variables).

- **addToCodeBody()**: adds the input string to the squashed code body. If a
class implements a translate function that wants to emit something to the
squashed code body, it must call this function with the code it wants to
emit. In case of loops, it automatically determines if the code needs to be
stored inside or outside the scope of that loop.

- **makeValidVarName()**: takes a string (e.g., a variable name) and converts
 it into a valid C++ variable name by replacing any forbidden characters with
 underscores.

- **buildArg()**: helps convert RooFit objects into arrays or other C++
representations for efficient computation.

- **addResult()**: adds (or overwrites) the string representing the result of
 a node.

> For each `translate()` function, it is important to call `addResult()` since
this is what enables the squashing to happen.

- **getResult()**: gets the result for the given node using the node name.
This node also performs the necessary code generation through recursive calls
 to `translate()`.

- **assembleCode()**: combines the generated code statements into the final
code body of the squashed function.

These functions will appear again in this document with more contextual
examples. For detailed in-line documentation (code comments), please see:

> [roofit/roofitcore/src/RooFit/CodegenContext.cxx](https://github.com/root-project/root/blob/master/roofit/roofitcore/src/RooFit/Detail/CodegenContext.cxx)


### b. RooFuncWrapper

> [roofit/roofitcore/inc/RooFuncWrapper.h](https://github.com/root-project/root/blob/master/roofit/roofitcore/inc/RooFuncWrapper.h)

This class wraps the generated C++ code in a RooFit object, so that it can be
 used like other RooFit objects.

It takes a function body as input and creates a callable function from it.
This allows users to evaluate the function and its derivatives efficiently.

#### Helper Functions

- **loadParamsAndData()** extracts parameters and observables from the
provided data and prepares them for evaluation.

- **declareAndDiffFunction()**: declare the function and create its
derivative.

- **gradient()**: calculates the gradient of the function with respect to its
 parameters.

- **buildCode()**: generates the optimized code for evaluating the function
and its derivatives.

- **dumpCode()**: prints the squashed code body to console (useful for
debugging).

- **dumpGradient()**: prints the derivative code body to console (useful for
debugging).

These functions will appear again in this document with more contextual
examples. For detailed in-line documentation (code comments), please see:

> [roofit/roofitcore/src/RooFuncWrapp9er.cxx](https://github.com/root-project/root/blob/master/roofit/roofitcore/src/RooFuncWrapper.cxx)


## Appendix C - Helper functions discussed in this document

- **RooFit::CodegenContext::addResult()**: For a specific class, it
will add whatever is represented on the right-hand side (a function call, an
expression, etc.) to the result of this class, which can then be propagated in
 the rest of the compute graph. A to call `addResult()`must be included in
`translate()` function.

  - Inputs: `key` (the name of the node to add the result for), `value` (the
    new name to assign/overwrite).
  - Output: Adds (or overwrites) the string representing the result of a node.

- **RooFit::CodegenContext::getResult()**: It helps lookup the
result of a child node (the string that the child node previously saved in a
variable using the `addResult()` function).

  - Input: `key` (the node to get the result string for).
  - Output: String representing the result of this node.

- **RooFit::CodegenContext::addToCodeBody()**: Takes whatever string
 is computed in its arguments and adds it to the overall function string (which
 will later be just-in-time compiled).

  - Inputs: `klass` (the class requesting this addition, usually 'this'), `in`
    (string to add to the squashed code).
  - Output: Adds the input string to the squashed code body.

- **RooFit::CodegenContext::addToGlobalScope()**: Helps declare and
initialize the results variable, so that it can be available globally
(throughout the function body).

  - Input: `str` (the string to add to the global scope).
  - Output: Adds the given string to the string block that will be emitted at
    the top of the squashed function.

- **RooFit::CodegenContext::assembleCode()**: combines the generated
code statements into the final code body of the squashed function.

  - Input: `returnExpr` (he string representation of what the squashed function
    should return, usually the head node).
  - Output: The final body of the function.

- **RooFit::CodegenContext::beginLoop()**: The code squashing task
will automatically build a For loop around the indented statements that follow
 this function.

  - Input: `in` (a pointer to the calling class, used to determine the loop
    dependent variables).
  - Output: A scope for iterating over vector observables.

- **RooFit::CodegenContext::buildArg()**: helps convert RooFit
objects into arrays or other C++ representations for efficient computation.

  - Input: `in` (the list to convert to array).
  - Output: Name of the array that stores the input list in the squashed code.

- **RooFit::CodegenContext::buildCall()**: Creates a string
representation of the function to be called and its arguments.

  - Input: A function with name `funcname`, passing some arguments.
  - Output: A string representation of the function to be called.

- **RooFit::Detail::makeValidVarName()**: It helps fetch and save a valid name
from the name of the respective RooFit class.

  - Input: `in` (the input string).
  - Output: A new string that is a valid variable name.

- **RooFuncWrapper::buildCode()**: Generates the optimized code for evaluating
 the function and its derivatives.

  - Input: `head` (starting mathematical expression).
  - Output: code for evaluating the function.

- **RooFuncWrapper::declareAndDiffFunction()**: Declare the function and create
its derivative.

  - Inputs: `funcName` (name of the function being differentiated), `funcBody`
 (actual mathematical formula or equation).
  - Output: Function declaration and its derivative.

- **RooFuncWrapper::dumpCode()**: Prints the squashed code body to console
(useful for debugging).

  - Output: Print squashed code body to console

- **RooFuncWrapper::dumpGradient()**: Prints the derivative code body to
console (useful for debugging).

  - Output: Print derivative code body to console.

- **RooFuncWrapper::gradient()**: Calculates the gradient of the function with
 respect to its parameters.

  - Input: `out` (array where the computed gradient values will be stored).
  - Output: Populates the `out` array with gradient values.

- **RooFuncWrapper::loadParamsAndData()** Extracts parameters and observables
from the provided data and prepares them for evaluation.

  - Input: `funcName` (function name), `head` (structure of the function),
    `paramSet` (input function's parameters), `data` (optional data points).
  - Output: Parameters, observables and other related information (e.g., output
    size of the provided function).



[^1]: For more details, please see this paper on [Automatic Differentiation of
Binned Likelihoods with Roofit and Clad]

[Automatic Differentiation of Binned Likelihoods with Roofit and Clad]: https://arxiv.org/abs/2304.02650

[^2]: For more details, please see this paper on [GPU Accelerated Automatic Differentiation with Clad]

[GPU Accelerated Automatic Differentiation with Clad]: https://arxiv.org/abs/2203.06139

[RooFit]: https://root.cern/manual/roofit/

[Clad]: https://compiler-research.org/clad/

[MathFuncs]: https://github.com/root-project/root/blob/master/roofit/roofitcore/inc/RooFit/Detail/MathFuncs.h

[MathFuncs]: https://github.com/root-project/root/blob/master/roofit/roofitcore/inc/RooFit/Detail/MathFuncs.h
