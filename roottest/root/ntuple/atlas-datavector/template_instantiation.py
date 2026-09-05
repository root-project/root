import pytest


class TestTemplateInstantiation:
    def test_instantiate_function_template(self):
        import ROOT

        ROOT.gInterpreter.Declare(
            r"""
            template <typename T> std::string foo() {
               int err = 0;
               char *demangled = TClassEdit::DemangleTypeIdName(typeid(T), err);
               std::string res{err == 0 ? demangled : ""};
               free(demangled);
               return res;
            }
            """
        )

        # The class type name as stored e.g. in an RNTuple schema: the second
        # (defaulted) template argument is stripped, according to the
        # KeepFirstTemplateArguments<1> rule in the dictionary selection.
        field_type_name = "AtlasLikeDataVector<CustomStruct>"

        # Instantiating with the shortened name works: resolving the class
        # name autoloads its dictionary, after which the compiler sees the
        # class template together with its default second template argument
        # and resolves the correct specialization.
        fully_qualified_type_name = "AtlasLikeDataVector<CustomStruct, DataModel_detail::NoBase>"
        assert ROOT.foo[field_type_name]() == fully_qualified_type_name

        # The first instantiation attempt has had the side effect of loading
        # the dictionary information for AtlasLikeDataVector, including the
        # alternative class type names, which e.g. the REntry pythonization
        # relies on as a fallback (see _try_getptr in _rntuple.py).
        alt_field_type_names = ROOT.TClassTable.GetClassAlternativeNames(field_type_name)

        assert len(alt_field_type_names) == 1
        assert alt_field_type_names[0] == fully_qualified_type_name

        # Instantiating with the fully qualified name works as well
        assert ROOT.foo[fully_qualified_type_name]() == fully_qualified_type_name


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
