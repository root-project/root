#pragma ifndef G__BOOL_H
#pragma define G__BOOL_H

#undef FALSE
#undef TRUE
#undef true
#undef false
enum bool { FALSE = 0, false = 0, TRUE = 1, true = 1 };
bool bool() { return false; }
//typedef int bool;
//const bool true=1;
//const bool TRUE=1;
//const bool false=0;
//const bool FALSE=0;

// This is not needed due to fix 1584
//#pragma link off class bool;
//#pragma link off function bool;

#pragma endif

