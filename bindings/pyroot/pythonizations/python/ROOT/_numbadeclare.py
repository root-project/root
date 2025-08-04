from cppyy import gbl as gbl_namespace


def _NumbaDeclareDecorator(input_types, return_type=None, name=None):
    """
    Decorator for making Python callables accessible in C++ by just-in-time compilation
    with numba and cling

    The decorator takes the given Python callable and just-in-time compiles (jits)
    wrapper functions with the given C++ types for input and return types. Eventually,
    the Python callable is accessible in the Numba namespace in C++.

    The implementation first jits with numba the Python callable. We support fundamental types and
    ROOT::VecOps::RVecs thereof. Note that you can get the jitted Python callable by the attribute
    numba_func. The C++ types are converted to the respective numba types and RVecs are accessible
    in Python by numpy arrays. After jitting the actual Python callable, we jit another Python wrapper,
    which converts the Python signature to a C-friendly signature. The wrapper code is accessible by
    the attribute __py_wrapper__. Next, the Python wrapper is given to cling to jit a C++ wrapper function,
    making the original Python callable accessible in C++. The wrapper code in C++ is accessible by
    the attribute __cpp_wrapper__.

    Note that the callable is fully compiled without side-effects. The numba jitting uses the nopython
    option which does not allow interaction with the Python interpreter. This means that you can use
    the resulting function also safely in multi-threaded environments.
    """
    # Make required imports
    try:
        import numba as nb
    except ImportError:
        raise Exception("Failed to import numba")
    try:
        import cffi
    except ImportError:
        raise Exception("Failed to import cffi")
    import re
    from typing import Union

    if hasattr(nb, "version_info") and nb.version_info >= (0, 54):
        pass

    CONTAINER_TYPES = {
        "RVec": {
            "match_pattern": r"RVec\w+|RVec<[\w\s]+>",
            "cpp_name": "ROOT::RVec",
        },
        "std::vector": {
            "match_pattern": r"std::vector<[\w\s]+>",
            "cpp_name": "std::vector",
        },
        "std::array": {
            "match_pattern": r"std::array<[\w\s,<>]+>",
            "cpp_name": "std::array",
        },
    }

    def get_container_type(cpp_type: str) -> Union[str, None]:
        return next((name for name in CONTAINER_TYPES if name in cpp_type), None)

    def is_container_type(cpp_type: str) -> bool:
        return get_container_type(cpp_type) is not None

    def get_container_cpp_name(cpp_type: str) -> Union[str, None]:
        container_type = get_container_type(cpp_type)
        if container_type is None:
            return None
        return CONTAINER_TYPES[container_type]["cpp_name"]

    # Normalize input types by stripping ROOT and VecOps namespaces from input types
    def normalize_typename(t):
        """
        Remove ROOT:: and VecOps:: namespaces
        """
        return t.replace("ROOT::", "").replace("VecOps::", "")

    input_types = [normalize_typename(t) for t in input_types]
    return_type = normalize_typename(return_type) if return_type is not None else None

    # Helper functions to determine types
    def get_inner_type(t: str, with_dims: bool = False) -> str:
        """
        Get inner typename of a templated C++ typename
        """
        try:
            if "<" not in t:
                # therefore, type must use a shorthand alias
                typemap = {
                    "F": "float",
                    "D": "double",
                    "I": "int",
                    "U": "unsigned int",
                    "L": "long",
                    "UL": "unsigned long",
                    "B": "bool",
                }
                rvec_start = t.find("RVec")  # discard a possible const modifier before "RVec"
                try:
                    return typemap[t[rvec_start + 4 :]]  # alias type characters come after "RVec"
                except KeyError:
                    raise Exception(
                        "Unrecognized type {}. Valid shorthand aliases of RVec are {}".format(
                            t, list("RVec" + i for i in typemap.keys())
                        )
                    )
            g = re.match("(.*)<(.*)>", t).groups(0)
            inner_type = g[1]
            # Handle std::array<T, N> by splitting the inner type, if requested
            if "," in inner_type and not with_dims:
                inner_type = inner_type.split(",")[0].strip()
            return inner_type
        except:  # noqa E722
            raise Exception("Failed to extract template argument of type {}".format(t))

    def get_numba_type(t):
        """
        Get numba type object from a C++ fundamental typename

        These are the types we use to jit the Python callable.
        """
        typemap = {
            "float": nb.float32,
            "double": nb.float64,
            "int": nb.int32,
            "unsigned int": nb.uint32,
            "long": nb.int64,
            "unsigned long": nb.uint64,
            "bool": nb.boolean,
        }
        if t in typemap:
            return typemap[t]
        raise Exception(
            "Type {} is not supported for jitting with numba. Valid fundamental types and container types (RVec, std::vector, std::array) thereof are {}".format(
                t, list(typemap.keys())
            )
        )

    def get_c_signature(input_types, return_type):
        """
        Get C friendly signature as numba type objects from C++ typenames

        We need the types to jit a Python wrapper, which can be accessed as a function pointer in C++.
        """
        c_input_types = []
        for t in input_types:
            if is_container_type(t):
                c_input_types += [nb.types.CPointer(get_numba_type(get_inner_type(t))), nb.int32]
            else:
                c_input_types.append(get_numba_type(t))
        if is_container_type(return_type):
            # We return an container through pointers as part of the input arguments. Note that the
            # pointer type in numba is always an int64 and is later on cast in C++ to the correct type.
            # In addition, we provide the size of the data type of the array for preallocating memory of
            # the returned array.
            # See the Python wrapper for further information why we are using these types.
            c_return_type = nb.void
            c_input_types += [
                nb.types.CPointer(nb.int64),  # Pointer to the data (the first element of the array)
                nb.types.CPointer(nb.int64),
            ]  # Size of the array in elements
        else:
            c_return_type = get_numba_type(return_type)
        return c_return_type, c_input_types

    def get_numba_signature(input_types, return_type):
        """
        Get numba signature as numba type objects from C++ typenames
        """
        nb_input_types = []
        for t in input_types:
            if is_container_type(t):
                nb_input_types.append(get_numba_type(get_inner_type(t))[:])
            else:
                nb_input_types.append(get_numba_type(t))
        if return_type is not None:
            if is_container_type(return_type):
                nb_return_type = get_numba_type(get_inner_type(return_type))[:]
            else:
                nb_return_type = get_numba_type(return_type)
        else:
            nb_return_type = None
        return nb_return_type, nb_input_types

    def add_container_input_type_ref(input_types_ref: list, const_mod: str, container_t: str) -> None:
        """
        Construct the type of a container input parameter for its use in the C++
        wrapper function signature.
        """
        container_base = get_container_type(container_t)
        cpp_name = get_container_cpp_name(container_t)
        full_type = container_t.replace(container_base, cpp_name, 1)
        tref = f"{const_mod}{full_type}&"
        input_types_ref.append(tref)

    def add_container_func_ptr_input_type(func_ptr_input_types: list, const_mod: str, container_t: str) -> None:
        """
        Construct the type of a container input parameter for its use in the cast
        of the function pointer of the jitted Python wrapper.
        """
        innert = get_inner_type(container_t)
        if innert == "bool":
            # Special treatment for bool: In numpy, bools have 1 byte
            innert = "char"

        func_ptr_input_types += ["{}{}*, int".format(const_mod, innert)]

    def inner(func, input_types=input_types, return_type=return_type, name=name):
        """
        Inner decorator without arguments, see outer decorator for documentation
        """

        # Jit the given Python callable with numba
        nb_return_type, nb_input_types = get_numba_signature(input_types, return_type)
        try:
            if nb_return_type is not None:
                nbjit = nb.jit(nb_return_type(*nb_input_types), nopython=True, inline="always")(func)
            else:
                nbjit = nb.jit(tuple(nb_input_types), nopython=True, inline="always")(func)
                nb_return_type = nbjit.nopython_signatures[-1].return_type
        except:  # noqa E722
            raise Exception("Failed to jit Python callable {} with numba.jit".format(func))
        func.numba_func = nbjit
        # return_type = "int"
        if return_type is None:
            type_map = {
                nb.types.boolean: "bool",
                nb.types.uint8: "unsigned int",
                nb.types.uint16: "unsigned int",
                nb.types.uint32: "unsigned int",
                nb.types.uint64: "unsigned long",
                nb.types.char: "int",
                nb.types.int8: "int",
                nb.types.int16: "int",
                nb.types.int32: "int",
                nb.types.int64: "long",
                nb.types.float32: "float",
                nb.types.float64: "double",
            }

            if nb_return_type in type_map:
                return_type = type_map[nb_return_type]
            elif "array" in nb.typeof(nb_return_type).name:
                return_type = "RVec<" + type_map[nb_return_type.dtype] + ">"
        # Create Python wrapper with C++ friendly signature

        # Define signature
        pywrapper_signature = [
            "ptr_{0}, size_{0}".format(i) if is_container_type(t) else "x_{}".format(i)
            for i, t in enumerate(input_types)
        ]
        if is_container_type(return_type):
            # If we return a container, we return via pointer the pointer of the allocated data,
            # the size in elements. In addition, we provide the size of the datatype in bytes.
            pywrapper_signature += ["ptrptr_r, ptrsize_r"]

        # Define arguments for jit function
        pywrapper_args_def = [
            "x_{0} = nb.carray(ptr_{0}, (size_{0},))".format(i) if is_container_type(t) else "x_{}".format(i)
            for i, t in enumerate(input_types)
        ]
        pywrapper_args = ["x_{}".format(i) for i in range(len(input_types))]

        # Define return operation
        if is_container_type(return_type):
            innert = get_inner_type(return_type)
            dtypesize = 1 if innert == "bool" else int(get_numba_type(innert).bitwidth / 8)
            pywrapper_return = "\n    ".join(
                [
                    "# Because we cannot manipulate the memory management of the numpy array we copy the data",
                    "ptr = malloc(r.size * {})".format(dtypesize),
                    "cp = nb.carray(ptr, r.size, dtype_r)",
                    "cp[:] = r[:]",
                    "# Return size of the array and the pointer to the copied data",
                    "ptrsize_r[0] = r.size",
                    "ptrptr_r[0] = cp.ctypes.data",
                ]
            )
        else:
            pywrapper_return = "return r"

        # Build wrapper code
        pywrappercode = '''\
def pywrapper({SIGNATURE}):
    """
    Wrapper function for the jitted Python callable with special treatment of arrays
    """
    # If a container is given, define numba carray wrapper for the input types
    {ARGS_DEF}
    # Call the jitted Python function
    r = nbjit({ARGS})
    # Return the result
    {RETURN}
        '''.format(
            SIGNATURE=", ".join(pywrapper_signature),
            ARGS_DEF="\n    ".join(pywrapper_args_def),
            ARGS=", ".join(pywrapper_args),
            RETURN=pywrapper_return,
        )

        glob = dict(globals())  # Make a shallow copy of the dictionary so we don't pollute the global scope
        glob["nb"] = nb
        glob["nbjit"] = nbjit

        ffi = cffi.FFI()
        ffi.cdef("void* malloc(long size);")
        C = ffi.dlopen(None)
        glob["malloc"] = C.malloc

        if is_container_type(return_type):
            glob["dtype_r"] = get_numba_type(get_inner_type(return_type))

        # Execute the pywrapper code and generate the wrapper function
        # which calls the jitted C function
        # Python 3.13 changes the semantics of the `locals()` builtin function
        # such that in optimized scopes (e.g. at function scope as it is
        # happening right now) the dictionary is not updated when changed. Bind
        # the `locals()` dictionary to a temporary object. This way, the call
        # to `exec()` will actually change the dictionary and the pywrapper
        # function will be found. Note that this change is backwards-compatible.
        local_objects = locals()
        exec(pywrappercode, glob, local_objects)

        if "pywrapper" not in local_objects:
            raise Exception("Failed to create Python wrapper function:\n{}".format(pywrappercode))

        # Jit the Python wrapper code
        c_return_type, c_input_types = get_c_signature(input_types, return_type)
        try:
            nbcfunc = nb.cfunc(c_return_type(*c_input_types), nopython=True)(local_objects["pywrapper"])
        except:  # noqa E722
            raise Exception("Failed to jit Python wrapper with numba.cfunc")
        func.__py_wrapper__ = pywrappercode
        func.__numba_cfunc__ = nbcfunc

        # Get address of jitted wrapper function
        address = nbcfunc.address

        # Infer name of the C++ wrapper function
        if not name:
            name = func.__name__

        # Build C++ wrapper for jitting with cling

        # Define:
        # - Input signature
        # - Function pointer types
        input_types_ref = []
        func_ptr_input_types = []
        for t in input_types:
            container_match_group = "|".join(entry["match_pattern"] for entry in CONTAINER_TYPES.values())
            m = re.match(rf"\s*(const\s+)?({container_match_group})", t)
            if m:
                const_mod = "" if m.group(1) is None else "const "
                container_t = m.group(2)

                add_container_input_type_ref(input_types_ref, const_mod, container_t)
                add_container_func_ptr_input_type(func_ptr_input_types, const_mod, container_t)
            else:
                input_types_ref.append(t)
                func_ptr_input_types.append(t)

        input_signature = ", ".join("{} x_{}".format(t, i) for i, t in enumerate(input_types_ref))

        if is_container_type(return_type):
            # See C++ wrapper code for the reason using these types
            innert = get_inner_type(return_type)
            func_ptr_input_types += ["{}**, long*".format("char" if innert == "bool" else innert)]
        func_ptr_type = "{RETURN_TYPE}(*)({INPUT_TYPES})".format(
            RETURN_TYPE="void*" if is_container_type(return_type) else return_type,
            INPUT_TYPES=", ".join(func_ptr_input_types),
        )

        # Define function call
        vecbool_conversion = []
        func_args = []
        for i, t in enumerate(input_types):
            if is_container_type(t):
                func_args += ["x_{0}.data(), x_{0}.size()".format(i)]
                if get_inner_type(t) == "bool":
                    # Copy the container<bool> to a container<char> to match the numpy memory layout
                    func_args[-1] = func_args[-1].replace("x_", "xb_")
                    vecbool_conversion += [f"{get_container_cpp_name(t)}<char> xb_{i} = x_{i};"]
            else:
                func_args += ["x_{}".format(i)]
        if is_container_type(return_type):
            # See C++ wrapper code for the reason using these arguments
            func_args += ["&ptr, &size"]

        # Define return operation
        if is_container_type(return_type):
            innert = get_inner_type(return_type)
            container_cpp = get_container_cpp_name(return_type)
            inner_with_dims = get_inner_type(return_type, with_dims=True)

            if innert == "bool":
                innert = "char"
                inner_with_dims = "char" + inner_with_dims[4:]

            # Default case: create container from pointer and size (e.g. std::vector, RVec)
            copy_lines = [f"{container_cpp}<{innert}> x_r(ptr, ptr + size);"]

            # Special case: if inner type includes dimensions (e.g. std::array<T, N>)
            # We construct the array then copy the data manually
            if innert != inner_with_dims:
                innert, dims = inner_with_dims.split(",")
                copy_lines = [f"{container_cpp}<{innert}, {dims}> x_r;std::copy(ptr, ptr + size, x_r.begin());"]

            return_op = "\n    ".join(
                [
                    "// Because a container type cannot take the ownership of external data, we have to copy the returned array",
                    "long size; // Size of the returned array",
                    f"{innert}* ptr; // Pointer to the data of the returned array",
                    f"funcptr({', '.join(func_args)});",
                    # TODO: Remove this copy as soon as the container type can adopt the ownership
                    *copy_lines,
                    "free(ptr);",
                    # If we return a container_type<bool>, we rely here on the automatic conversion of container_type<char> to container_type<bool>
                    "return x_r;",
                ]
            )
        else:
            return_op = "return funcptr({});".format(", ".join(func_args))

        # Build wrapper code
        cppwrappercode = """\
namespace Numba {{
/*
 * C++ wrapper function around the jitted Python wrapping which calls the jitted Python callable
 */
{RETURN_TYPE} {FUNC_NAME}({INPUT_SIGNATURE}) {{
    // Create a function pointer from the jitted Python wrapper
    const auto funcptr = reinterpret_cast<{FUNC_PTR_TYPE}>({FUNC_PTR});
    // Perform conversion of container_type<bool>
    {VECBOOL_CONVERSION}
    // Return the result
    {RETURN_OP}
}}
}}""".format(
            RETURN_TYPE="ROOT::" + return_type if "RVec" in return_type else return_type,
            FUNC_NAME=name,
            INPUT_SIGNATURE=input_signature,
            FUNC_PTR=address,
            FUNC_PTR_TYPE=func_ptr_type,
            VECBOOL_CONVERSION="\n    ".join(vecbool_conversion),
            RETURN_OP=return_op,
        )

        # Jit wrapper C++ code
        err = gbl_namespace.gInterpreter.Declare(cppwrappercode)
        if not err:
            raise Exception("Failed to jit C++ wrapper code with cling:\n{}".format(cppwrappercode))
        func.__cpp_wrapper__ = cppwrappercode

        return func

    return inner
