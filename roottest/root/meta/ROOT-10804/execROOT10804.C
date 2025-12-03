int numErrors = 0;

void check(const char *what) {
  if (!TClassTable::GetProto(what)) {
    std::cerr << "FAILED to get TProtoClass for \"" << what << "\"\n";
    ++numErrors;
  }
}

void check_not(const char *what) {
  if (TClassTable::GetProto(what)) {
    std::cerr << "FAILED to NOT get TProtoClass for \"" << what
              << "\" (expected nullptr but got != nullptr)\n";
    ++numErrors;
  }
}

int execROOT10804() {
  if (gSystem->Load("libROOT10804_dictrflx") < 0) {
    std::cerr << "Failed to load ROOT10804Dict!\n";
    return 1;
  }
  check("Outer::Inline::Class");
  check("Outer::Class");

  check("list<Outer::Inline::Class>");
  check("list<Outer::Class>");
  check("std::list<Outer::Inline::Class>");
  check("std::list<Outer::Class>");

  check("Outer::Container<Outer::Inline::Class>");
  check("Outer::Container<Outer::Inline::Class>");
  check("Outer::Container<Outer::Class>");

  check("Outer::Inline::Template<Outer::Inline::Class>");
  check("Outer::Template<Outer::Inline::Class>");
  check("Outer::Inline::Template<Outer::Class>");
  check("Outer::Template<Outer::Class>");

  check("Outer::Container<"
          "Outer::Inline::Template<"
            "Outer::Inline::Class>>");

  check("Outer::Container<"
          "Outer::Inline::Template<"
            "Outer::Inline::Class>>>");

  check("Outer::Container<"
          "Outer::Inline::Template<"
            "Outer::Class>>");
  check("Outer::Container<"
          "Outer::Template<"
            "Outer::Inline::Class>>");

  check("Outer::Container<"
          "Outer::Template<"
            "Outer::Class>>");

  // Container is not Inline::
  check_not("Outer::Inline::Container<"
              "Outer::Inline::Template<"
                "Outer::Inline::Class>>");

  // Float16_t won't get a demangled alternate name,
  // only as-written-in-selection.xml
  check("Outer::Template<Float16_t>");
  // FIXME: can be resolved now that the payload has been parsed.
  // And that shows that the test needs to check *one* type name only,
  // before parsing the payload.
  //!!! check_not("Outer::Inline::Template<Float16_t>");

  // Double32_t won't get a demangled alternate name,
  // only as-written-in-selection.xml
  check("Outer::Container<Outer::Template<Double32_t>>");
  // FIXME: can be resolved now that the payload has been parsed.
  // And that shows that the test needs to check *one* type name only,
  // before parsing the payload.
  //!!! check_not("Outer::Container<Outer::Inline::Template<Double32_t>>");

  // Cross-check: not selected.
  check_not("Outer::Container<int>");
  check_not("Outer::Container<Outer::Template<Float16_t>>");
  check_not("Outer::Template<float>"); // but did select Template<Float16_t>!
  return numErrors ? 1 : 0;
}
