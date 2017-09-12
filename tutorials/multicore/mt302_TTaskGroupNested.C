/// \file
/// \ingroup tutorial_multicore
/// \notebook
/// Calculate Fibonacci numbers exploiting nested parallelism through TTaskGroup.
///
/// \macro_code
///
/// \date August 2017
/// \author Danilo Piparo

int Fibonacci(int n) {
   if( n<2 ) {
      return n;
   } else {
      int x, y;
      ROOT::Experimental::TTaskGroup tg;
      tg.Run([&]{x=Fibonacci(n-1);});
      tg.Run([&]{y=Fibonacci(n-2);});
      tg.Wait();
      return x+y;
   }
}

void mt302_TTaskGroupNested()
{

   ROOT::EnableImplicitMT(4);

   cout << "Fibonacci(33) = " << Fibonacci(33) << endl;

}

