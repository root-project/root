
#include "saxParserSimpleHandle.h"

int execSaxParserSimple()
{
   const Int_t nIterations = 1000;
   TSAXParser *saxParser = new TSAXParser();
   SAXHandler *saxHandler = new SAXHandler();
   TStopwatch timer;
   saxParser->ConnectToHandler("SAXHandler", saxHandler);
   timer.Start();
   for (Int_t i = 0; i < nIterations; i++) {
      saxParser->ParseFile("./saxSimpleExample.xml");
      saxHandler->Quiet();
   }

   auto realTime = timer.RealTime();
   if (realTime > 15)
      std::cout << "WARNING: The parsing took " << realTime << " seconds. This may be too much\n";

   return 0;
}
