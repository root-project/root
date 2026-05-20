#include "TNtuple.h"
#include "TSQLRow.h"
#include "TSQLResult.h"

int execQuery() 
{
   TNtuple *ed = new TNtuple("ed","ed","pt:val:stat");
   ed->SetScanField(0);
   ed->Fill(1.0, 5.0, 4.0);
   TSQLRow *row = ed->Query("pt:val:stat")->Next();
   fprintf(stdout,"1: %s 2: %s 3: %s\n",row->GetField(0),row->GetField(1),row->GetField(2));
   return 0;
}


