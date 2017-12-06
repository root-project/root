/// \file
/// \ingroup tutorial_multicore
/// \notebook -js
/// Shows how to run items of work asynchronously with a TTaskGroup.
///
/// \macro_code
///
/// \date August 2017
/// \author Danilo Piparo

void workItem0()
{
   printf("Running workItem0...\n");
}

void mt301_TTaskGroupSimple()
{

   ROOT::EnableImplicitMT(4);

   // Create the task group and give work to it
   ROOT::Experimental::TTaskGroup tg;

   tg.Run(workItem0);
   tg.Run([]() { printf("Running workItem1...\n"); });

   printf("Running something in the \"main\" thread\n");

   // Wait until all items are complete
   tg.Wait();

   printf("All work completed.\n");
}
