import pytest


class TestTemplateInstantiation:
    def test_instantiate_function_template(self):
        import ROOT

        ROOT.gInterpreter.Declare(r"template <typename T> void foo() {}")

        # The call raises an exception that the function cannot be found.
        # What is really happening is that a template instantiation is tried
        # with the "wrong" signature, as the class type passed by the user is
        # not what the compiler sees.
        field_type_name = "AtlasLikeDataVector<CustomStruct>"
        with pytest.raises(TypeError):
            ROOT.foo[field_type_name]()

        # The first attempt at instantiating the template has had the side
        # effect of loading the dictionary information for AtlasLikeDataVector,
        # including the alternative class type names
        alt_field_type_names = ROOT.TClassTable.GetClassAlternativeNames(field_type_name)

        fully_qualified_type_name = "AtlasLikeDataVector<CustomStruct, DataModel_detail::NoBase>"

        assert len(alt_field_type_names) == 1
        assert alt_field_type_names[0] == fully_qualified_type_name

        # Retrying the instantiation with the alternative name should work
        ROOT.foo[alt_field_type_names[0]]()


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
