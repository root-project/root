// Test string parsing with TXMLEngine
// Also keep file include via ENTITY
// Author: Sergey Linev   7.04.2026

#include "TXMLEngine.h"

void runenginebad()
{
   // First create engine
   TXMLEngine xml;

   // Try to include file using path via parent directory
   // File in principle exists but in generally it is not allowed to navigate via parent path
   XMLDocPointer_t xmldoc1 = xml.ParseString(
    "<?xml version=\"1.0\"?>\n"
    "   <!DOCTYPE main [\n"
    "     <!ENTITY file_in_parent_dir SYSTEM \"../include.xml\">\n"
    "   ]>\n"
    "<main>&file_in_parent_dir;</main>");
   if (!xmldoc1) {
      std::cout << "Include from parent directory is not allowed" << std::endl;
   } else {
      std::cerr << "FAILURE - parsing must not work when parent directory is specified" << std::endl;
      xml.FreeDoc(xmldoc1);
   }

   XMLDocPointer_t xmldoc2 = xml.ParseString(
    "<?xml version=\"1.0\"?>\n"
    "   <!DOCTYPE main [\n"
    "     <!ENTITY file_in_parents_dir SYSTEM \"path/inside/../../../../core/include.xml\">\n"
    "   ]>\n"
    "<main>&file_in_parents_dir;</main>");
   if (!xmldoc2) {
      std::cout << "Include from parent directory is not allowed" << std::endl;
   } else {
      std::cerr << "FAILURE - parsing must not work when parent directory is specified" << std::endl;
      xml.FreeDoc(xmldoc2);
   }

   // Try to include file using global path
   // Should be refused before trying to open such file so checking proper error message as well
#ifdef R__WIN32
   XMLDocPointer_t xmldoc3 = xml.ParseString(
    "<?xml version=\"1.0\"?>\n"
    "   <!DOCTYPE main [\n"
    "     <!ENTITY file_from_top_dir SYSTEM \"c:/Windows/System32/sti.dll\">\n"
    "  ]>\n"
    "<main>&file_from_top_dir;</main>");
#else
   XMLDocPointer_t xmldoc3 = xml.ParseString(
    "<?xml version=\"1.0\"?>\n"
    "   <!DOCTYPE main [\n"
    "     <!ENTITY file_from_top_dir SYSTEM \"/usr/lib/firewalld/services/irc.xml\">\n"
    "   ]>\n"
    "<main>&file_from_top_dir;</main>");
#endif
   if (!xmldoc3) {
      std::cout << "Include from top directory is not allowed" << std::endl;
   } else {
      std::cerr << "FAILURE - parsing must not work when top directory is specified" << std::endl;
      xml.FreeDoc(xmldoc3);
   }
}
