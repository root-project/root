#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

int iostream_state()
{
   ifstream f1("iostream_state.dat", ios_base::in);
   string line;
   istringstream buf;
   int i = 0;
   int val = 0.0F;
   while (getline(f1, line)) {
      buf.str(line);
      cout << "line: " << line << endl;
      while (buf >> val) {
         cout << i
              << ": "
              << val
              << " state: "
              << buf.eof()
              << ':'
              << buf.fail()
              << ':'
              << buf.bad()
              << endl;
         if (i != 9) {
            if (buf.eof() || buf.fail() || buf.bad()) {
               f1.close();
               return 1;
            }
         }
         else {
            if (!buf.eof() || buf.fail() || buf.bad()) {
               f1.close();
               return 1;
            }
         }
         ++i;
      }
      cout << "end loop: "
           << i
           << ": "
           << val
           << " state: "
           << buf.eof()
           << ':'
           << buf.fail()
           << ':'
           << buf.bad()
           << endl;
      if (!buf.eof() || !buf.fail() || buf.bad()) {
         f1.close();
         return 1;
      }
      buf.clear();
   }
   f1.close();
   return 0;
}

int main()
{
   int err = iostream_state();
   if (err) {
      cout << "Invalid iostream state seen!" << endl;
   }
   return err;
}

