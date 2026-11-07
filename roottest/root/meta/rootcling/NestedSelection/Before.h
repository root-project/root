// Parsing this selection-only header in rootcling is intentional. The guard
// detects an unwanted direct include during later dictionary C++ compilation.
#ifndef __ROOTCLING__
#error "Headers collected before a nested LinkDef must not be emitted directly."
#endif
