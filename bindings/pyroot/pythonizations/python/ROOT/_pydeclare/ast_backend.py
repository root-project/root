# Author: Jonas Rembser CERN 08/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

"""The AST-traversal backend.

Walks the Python syntax tree of the decorated callable and emits C++.  Because
the argument types are known -- RDataFrame knows its column types, and the
decorator takes them explicitly -- forward type inference through the tree is
enough to choose the right lowering for every operation.

Compared to the tracing backend this one sees the *source*, so it can translate
control flow: if/elif/else, for, while, break/continue and early returns.
"""

import ast
import inspect
import textwrap

from . import cpptypes as ct
from . import emit
from .cpptypes import UNKNOWN_T
from .errors import PyDeclareError, raise_unsupported

CPP_KEYWORDS = frozenset(
    """
    alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t
    char32_t class compl concept const consteval constexpr constinit const_cast continue co_await
    co_return co_yield decltype default delete do double dynamic_cast else enum explicit export
    extern false float for friend goto if inline int long mutable namespace new noexcept not not_eq
    nullptr operator or or_eq private protected public register reinterpret_cast requires return
    short signed sizeof static static_assert static_cast struct switch template this thread_local
    throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while
    xor xor_eq NULL
    """.split()
)

BUILTIN_CASTS = {"float", "int", "bool"}

#: Builtins that are safe to run at declaration time when folding constants.
PURE_BUILTINS = frozenset(
    ["abs", "bool", "float", "int", "len", "max", "min", "pow", "round", "sum", "range", "list", "tuple"]
)


class _ModuleRef:
    """A reference to numpy or math in the callable's enclosing scope."""

    def __init__(self, kind, path):
        self.kind = kind  # "numpy" or "math"
        self.path = path


class _FuncRef:
    """A numpy/math function that has a C++ counterpart."""

    def __init__(self, name, path):
        self.name = name
        self.path = path


class _CppRef:
    """A cppyy entity (class, namespace or free function) usable from C++."""

    def __init__(self, cpp_name):
        self.cpp_name = cpp_name


class _DeclaredRef:
    """Another callable that was already translated by this package."""

    def __init__(self, cpp_name, return_type):
        self.cpp_name = cpp_name
        self.return_type = return_type


class _Scope:
    """One lexical block; variables declared here die at the closing brace."""

    def __init__(self, parent=None):
        self.parent = parent
        self.names = {}

    def lookup(self, name):
        scope = self
        while scope is not None:
            if name in scope.names:
                return scope.names[name]
            scope = scope.parent
        return None


class AstTranspiler:
    """Translate one Python callable into the body of a C++ function."""

    def __init__(self, func, param_names, param_types, declared_return, cpp_param_names):
        self.func = func
        self.param_names = param_names
        self.param_types = param_types
        self.declared_return = declared_return
        self.cpp_param_names = cpp_param_names

        self.lines = []
        self.indent = 1
        self.depth = 0
        self.tmp_counter = 0
        self.flat_scope = False
        self.hoisted = {}  # python name -> (cpp name, CppType)
        self.collected = {}  # python name -> CppType, filled by the first pass
        self.collected_depth = {}  # python name -> set of block depths it is assigned at
        self.return_types = []
        self.loop_depth = 0
        self.globals_cache = None

    # -- error helpers ------------------------------------------------------

    def fail_at(self, node):
        def _fail(msg, hint=None):
            raise_unsupported(self.func, node, msg, hint)

        return _fail

    # -- name helpers -------------------------------------------------------

    def cpp_name(self, name):
        if name in CPP_KEYWORDS or name.startswith("PyD"):
            return name + "_"
        return name

    def tmp(self, stem="t"):
        self.tmp_counter += 1
        return "_pyd_{}{}".format(stem, self.tmp_counter)

    def write(self, text):
        self.lines.append("   " * self.indent + text)

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    def translate(self):
        tree = self._parse()
        body = tree.body

        # Pass 1: a flat scope, so that every read resolves; this only collects
        # the types of variables that will have to be hoisted to function scope.
        self.flat_scope = True
        saved = (self.lines, self.indent)
        self.lines = []
        self._run(body)
        self.lines, self.indent = saved

        hoist_names = {n for n, depths in self.collected_depth.items() if any(d > 0 for d in depths)}

        # Pass 2: the real one.
        self.flat_scope = False
        self.return_types = []
        self.tmp_counter = 0
        self.hoisted = {}
        for name in sorted(hoist_names):
            t = self.collected.get(name, UNKNOWN_T)
            if t is None or t.is_unknown:
                raise PyDeclareError(
                    "Cannot determine the C++ type of the variable '{}', which is assigned inside a "
                    "nested block and therefore has to be declared up front.\n"
                    "  Give it a type by annotating the assignment, for example "
                    "\"{}: 'double' = 0.0\", or assign it once before the block.".format(name, name)
                )
            self.hoisted[name] = (self.cpp_name(name), t)
            self.write("{} {}{{}};".format(t.cpp(), self.cpp_name(name)))

        self._run(body)

        if not self.return_types:
            raise PyDeclareError(
                "The callable '{}' has no reachable 'return' statement; "
                "a translated function must return a value.".format(getattr(self.func, "__name__", "<lambda>"))
            )

        return "\n".join(self.lines), self._deduce_return_type()

    def _deduce_return_type(self):
        if self.declared_return is not None:
            return self.declared_return
        types = [t for t in self.return_types if t is not None and not t.is_unknown]
        if len(types) == len(self.return_types) and types and all(t == types[0] for t in types):
            return types[0]
        return None  # emit 'auto' and let the compiler deduce it

    def _run(self, body):
        self.scope = _Scope()
        for i, pname in enumerate(self.param_names):
            self.scope.names[pname] = emit.Value(self.cpp_param_names[i], self.param_types[i])
        # Variables hoisted to function scope are visible from the first line on,
        # which is what makes Python's function-wide variable scope work here.
        for pname, (cpp, type_) in self.hoisted.items():
            self.scope.names[pname] = emit.Value(cpp, type_)
        self.exec_block(body, new_scope=False)

    def _parse(self):
        try:
            src = inspect.getsource(self.func)
        except (OSError, TypeError):
            raise PyDeclareError(
                "Cannot read the source of the callable. The AST backend needs the Python source; "
                "callables defined in the interpreter prompt or in an exec() string cannot be "
                "translated. Use the tracing backend, or move the function into a file."
            )
        src = textwrap.dedent(src)
        try:
            mod = ast.parse(src)
        except SyntaxError as exc:
            raise PyDeclareError("Cannot parse the source of the callable: {}".format(exc))

        node = mod.body[0]
        if isinstance(node, ast.FunctionDef):
            self._check_args(node.args)
            return node
        # A decorated lambda arrives as an assignment or a bare expression.
        for sub in ast.walk(mod):
            if isinstance(sub, ast.Lambda):
                self._check_args(sub.args)
                return ast.Module(body=[ast.Return(value=sub.body)], type_ignores=[])
        raise PyDeclareError("Expected a function definition or a lambda, got {}".format(type(node).__name__))

    def _check_args(self, args):
        if args.vararg is not None or args.kwarg is not None:
            raise PyDeclareError("*args and **kwargs are not supported")
        if args.kwonlyargs:
            raise PyDeclareError("Keyword-only arguments are not supported")
        names = [a.arg for a in args.posonlyargs] + [a.arg for a in args.args]
        if names != list(self.param_names):
            raise PyDeclareError(
                "The declared signature does not match the callable's parameters: expected {}, got {}".format(
                    list(self.param_names), names
                )
            )

    # -----------------------------------------------------------------------
    # Statements
    # -----------------------------------------------------------------------

    def exec_block(self, body, new_scope=True):
        if new_scope and not self.flat_scope:
            self.scope = _Scope(self.scope)
        for stmt in body:
            self.exec_stmt(stmt)
        if new_scope and not self.flat_scope:
            self.scope = self.scope.parent

    def exec_stmt(self, node):
        handler = getattr(self, "st_" + type(node).__name__, None)
        if handler is None:
            raise_unsupported(
                self.func,
                node,
                "The statement '{}' is not supported".format(type(node).__name__),
                hint="Supported statements are: return, assignment (including augmented and "
                "annotated), if/elif/else, for, while, break, continue, pass, assert and "
                "expression statements.",
            )
        handler(node)

    def st_Pass(self, node):
        pass

    def st_Expr(self, node):
        # Docstrings and other bare constants are dropped.
        if isinstance(node.value, ast.Constant):
            return
        value = self.expr(node.value)
        self.write("{};".format(value.code))

    def st_Return(self, node):
        if node.value is None:
            raise_unsupported(self.func, node, "'return' without a value is not supported")
        value = self.expr(node.value)
        self.return_types.append(value.type)
        if self.declared_return is not None:
            self.write("return {};".format(self._converted(value, self.declared_return)))
        else:
            self.write("return {};".format(value.code))

    def st_Assign(self, node):
        if len(node.targets) != 1:
            raise_unsupported(self.func, node, "Chained assignment (a = b = c) is not supported")
        self._assign_to(node.targets[0], self.expr(node.value), node)

    def st_AnnAssign(self, node):
        if node.value is None:
            raise_unsupported(self.func, node, "A type annotation without a value is not supported")
        if not isinstance(node.target, ast.Name):
            raise_unsupported(self.func, node, "Annotated assignment is only supported for plain variables")
        spelling = self._annotation_text(node.annotation)
        declared = ct.parse_type(spelling)
        value = self.expr(node.value)
        name = node.target.id
        cpp = self.cpp_name(name)
        self._record(name, declared)
        if name in self.hoisted:
            self.write("{} = {};".format(cpp, self._converted(value, declared)))
        else:
            existing = self.scope.lookup(name)
            if existing is not None:
                raise_unsupported(self.func, node, "Variable '{}' is already declared".format(name))
            self.write("{} {} = {};".format(declared.cpp(), cpp, self._converted(value, declared)))
            self.scope.names[name] = emit.Value(cpp, declared)

    def _annotation_text(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        raise_unsupported(
            self.func,
            node,
            "Type annotations must be C++ type names given as strings",
            hint="For example: x: 'double' = 0.0",
        )

    def st_AugAssign(self, node):
        op = _BINOP_NAMES.get(type(node.op).__name__)
        if op is None:
            raise_unsupported(self.func, node, "Augmented assignment with this operator is not supported")
        current = self.expr(_as_load(node.target))
        result = emit.binop(op, current, self.expr(node.value), self.fail_at(node))
        self._assign_to(node.target, result, node)

    def _record(self, name, type_):
        """Remember an assignment for the hoisting pass."""
        self.collected_depth.setdefault(name, set()).add(self.depth)
        prev = self.collected.get(name)
        if prev is None:
            self.collected[name] = type_
        elif prev != type_:
            # Two different types for one name: keep it unknown so that the
            # hoisting pass reports it rather than picking one at random.
            self.collected[name] = UNKNOWN_T if (prev.is_unknown or type_.is_unknown) else _unify(prev, type_)

    def _converted(self, value, target_type):
        """The expression, converted to target_type unless it already has it."""
        if value.type == target_type:
            return value.code
        return "{}::Return<{}>({})".format(emit.PYD, target_type.cpp(), value.code)

    def _assign_to(self, target, value, node):
        if isinstance(target, ast.Subscript):
            obj = self.expr(target.value)
            index = self.expr(target.slice)
            slot = emit.subscript(obj, index, self.fail_at(node))
            self.write("{} = {};".format(slot.code, value.code))
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            raise_unsupported(self.func, node, "Tuple unpacking is not supported")
        if not isinstance(target, ast.Name):
            raise_unsupported(self.func, node, "Cannot assign to {}".format(type(target).__name__))

        name = target.id
        cpp = self.cpp_name(name)
        self._record(name, value.type)

        if name in self.hoisted:
            declared = self.hoisted[name][1]
            self.write("{} = {};".format(cpp, self._converted(value, declared)))
            return

        existing = self.scope.lookup(name)
        if existing is None:
            self.write("auto {} = {};".format(cpp, value.code))
            self.scope.names[name] = emit.Value(cpp, value.type)
        else:
            if not (existing.type.is_unknown or value.type.is_unknown) and existing.type != value.type:
                raise_unsupported(
                    self.func,
                    node,
                    "Variable '{}' changes type from '{}' to '{}'".format(name, existing.type, value.type),
                    hint="A translated variable keeps the type it was first assigned; C++ has no "
                    "dynamically typed variables. Use a second variable instead.",
                )
            self.write("{} = {};".format(cpp, value.code))

    def st_If(self, node):
        cond = emit.truth(self.expr(node.test), self.fail_at(node))
        self.write("if ({}) {{".format(cond.code))
        self.indent += 1
        self.depth += 1
        self.exec_block(node.body)
        self.depth -= 1
        self.indent -= 1
        if node.orelse:
            self.write("} else {")
            self.indent += 1
            self.depth += 1
            self.exec_block(node.orelse)
            self.depth -= 1
            self.indent -= 1
        self.write("}")

    def st_While(self, node):
        if node.orelse:
            raise_unsupported(self.func, node, "'while ... else' is not supported")
        cond = emit.truth(self.expr(node.test), self.fail_at(node))
        self.write("while ({}) {{".format(cond.code))
        self.indent += 1
        self.depth += 1
        self.loop_depth += 1
        self.exec_block(node.body)
        self.loop_depth -= 1
        self.depth -= 1
        self.indent -= 1
        self.write("}")

    def st_For(self, node):
        if node.orelse:
            raise_unsupported(self.func, node, "'for ... else' is not supported")
        if not isinstance(node.target, ast.Name):
            raise_unsupported(
                self.func,
                node,
                "Only a single loop variable is supported",
                hint="enumerate() and zip() are not translated; loop over an index range instead.",
            )
        name = node.target.id
        cpp = self.cpp_name(name)

        if isinstance(node.iter, ast.Call) and self._is_range(node.iter.func):
            self._for_range(node, name, cpp)
            return

        iterable = self.expr(node.iter)
        if not iterable.type.is_container and not iterable.type.is_unknown:
            raise_unsupported(
                self.func,
                node,
                "Cannot iterate over a value of type '{}'".format(iterable.type),
                hint="Only arrays and range() can be iterated.",
            )
        elem = iterable.type.scalar()
        self._record(name, elem)
        self.write("for (auto {} : {}) {{".format(cpp, iterable.code))
        self._loop_body(node, name, cpp, elem)

    def _is_range(self, func):
        return isinstance(func, ast.Name) and func.id == "range" and self.scope.lookup("range") is None

    def _for_range(self, node, name, cpp):
        args = [self.expr(a) for a in node.iter.args]
        if node.iter.keywords or not 1 <= len(args) <= 3:
            raise_unsupported(self.func, node, "range() takes one to three positional arguments")
        start = emit.Value("0", ct.LONG_T) if len(args) == 1 else args[0]
        stop = args[0] if len(args) == 1 else args[1]
        step = args[2] if len(args) == 3 else emit.Value("1", ct.LONG_T)

        if step.code.lstrip("-").isdigit():
            step_value = int(step.code)
        else:
            raise_unsupported(
                self.func,
                node,
                "The step of range() must be an integer literal",
                hint="A runtime step would need a direction check on every iteration; write two "
                "loops, or use a while loop.",
            )
        if step_value == 0:
            raise_unsupported(self.func, node, "range() step must not be zero")

        bound = self.tmp("stop")
        self.write("{{ const long {} = static_cast<long>({});".format(bound, stop.code))
        self.indent += 1
        cmp_op = "<" if step_value > 0 else ">"
        incr = (
            "{} += {}".format(cpp, step_value)
            if abs(step_value) != 1
            else ("++" + cpp if step_value > 0 else "--" + cpp)
        )
        self.write(
            "for (long {} = static_cast<long>({}); {} {} {}; {}) {{".format(cpp, start.code, cpp, cmp_op, bound, incr)
        )
        self._record(name, ct.LONG_T)
        self._loop_body(node, name, cpp, ct.LONG_T)
        self.indent -= 1
        self.write("}")

    def _loop_body(self, node, name, cpp, elem_type):
        self.indent += 1
        self.depth += 1
        self.loop_depth += 1
        if not self.flat_scope:
            self.scope = _Scope(self.scope)
        self.scope.names[name] = emit.Value(cpp, elem_type)
        for stmt in node.body:
            self.exec_stmt(stmt)
        if not self.flat_scope:
            self.scope = self.scope.parent
        self.loop_depth -= 1
        self.depth -= 1
        self.indent -= 1
        self.write("}")

    def st_Break(self, node):
        if not self.loop_depth:
            raise_unsupported(self.func, node, "'break' outside a loop")
        self.write("break;")

    def st_Continue(self, node):
        if not self.loop_depth:
            raise_unsupported(self.func, node, "'continue' outside a loop")
        self.write("continue;")

    def st_Assert(self, node):
        cond = emit.truth(self.expr(node.test), self.fail_at(node))
        message = "assertion failed in {}".format(getattr(self.func, "__name__", "<lambda>"))
        if node.msg is not None and isinstance(node.msg, ast.Constant) and isinstance(node.msg.value, str):
            message = node.msg.value
        self.write('if (!({})) {{ throw std::runtime_error("{}"); }}'.format(cond.code, message.replace('"', '\\"')))

    # -----------------------------------------------------------------------
    # Expressions
    # -----------------------------------------------------------------------

    def expr(self, node):
        """Translate an expression node, or fail loudly."""
        # An expression that does not depend on any argument is a compile-time
        # constant: evaluate it in Python and inline the result, exactly like
        # numba freezes globals.
        folded = self._try_fold(node)
        if folded is not None:
            return folded

        handler = getattr(self, "ex_" + type(node).__name__, None)
        if handler is None:
            raise_unsupported(self.func, node, "The expression '{}' is not supported".format(type(node).__name__))
        result = handler(node)
        if not isinstance(result, emit.Value):
            raise_unsupported(
                self.func,
                node,
                "'{}' cannot be used as a value here".format(ast.dump(node)[:60]),
            )
        return result

    def ex_Constant(self, node):
        return emit.literal(node.value, self.fail_at(node))

    def ex_Name(self, node):
        local = self.scope.lookup(node.id)
        if local is not None:
            return local
        resolved = self._resolve_global(node.id, node)
        if isinstance(resolved, emit.Value):
            return resolved
        return resolved  # a reference object; the caller decides what to do

    def ex_BinOp(self, node):
        op = _BINOP_NAMES.get(type(node.op).__name__)
        if op is None:
            raise_unsupported(self.func, node, "Binary operator not supported")
        return emit.binop(op, self.expr(node.left), self.expr(node.right), self.fail_at(node))

    def ex_UnaryOp(self, node):
        op = {"UAdd": "+", "USub": "-", "Not": "not", "Invert": "~"}.get(type(node.op).__name__)
        if op is None:
            raise_unsupported(self.func, node, "Unary operator not supported")
        return emit.unaryop(op, self.expr(node.operand), self.fail_at(node))

    def ex_BoolOp(self, node):
        op = "and" if isinstance(node.op, ast.And) else "or"
        return emit.boolop(op, [self.expr(v) for v in node.values], self.fail_at(node))

    def ex_Compare(self, node):
        fail = self.fail_at(node)
        ops = [_CMP_NAMES.get(type(o).__name__) for o in node.ops]
        if any(o is None for o in ops):
            raise_unsupported(self.func, node, "Comparison operator not supported")
        operands = [self.expr(node.left)] + [self.expr(c) for c in node.comparators]
        if len(ops) == 1:
            return emit.compare(ops[0], operands[0], operands[1], fail)
        if any(v.type.is_container for v in operands):
            raise_unsupported(
                self.func,
                node,
                "Chained comparisons are not supported for arrays",
                hint="numpy raises here too; combine two comparisons with '&'.",
            )
        parts = [emit.compare(ops[i], operands[i], operands[i + 1], fail) for i in range(len(ops))]
        return emit.boolop("and", parts, fail)

    def ex_IfExp(self, node):
        return emit.ternary(self.expr(node.test), self.expr(node.body), self.expr(node.orelse), self.fail_at(node))

    def ex_Subscript(self, node):
        obj = self.expr(node.value)
        if isinstance(node.slice, ast.Slice):
            sl = node.slice
            start = self.expr(sl.lower) if sl.lower is not None else None
            stop = self.expr(sl.upper) if sl.upper is not None else None
            step = self.expr(sl.step) if sl.step is not None else None
            return emit.slice_(obj, start, stop, step, self.fail_at(node))
        if isinstance(node.slice, ast.Tuple):
            raise_unsupported(self.func, node, "Multi-dimensional indexing is not supported")
        return emit.subscript(obj, self.expr(node.slice), self.fail_at(node))

    def ex_Attribute(self, node):
        base = self._maybe_ref(node.value)
        name = node.attr

        if isinstance(base, _ModuleRef):
            return self._module_attribute(base, name, node)
        if isinstance(base, _CppRef):
            return _CppRef("{}::{}".format(base.cpp_name, name))
        if isinstance(base, _DeclaredRef):
            raise_unsupported(self.func, node, "Cannot take an attribute of a declared function")

        value = base
        if value.type.is_container:
            if name in emit.ARRAY_METHODS:
                return _BoundMethod(value, name)
            if name == "size":
                return emit.length(value, self.fail_at(node))
            if name == "astype":
                return _BoundMethod(value, "astype")
            raise_unsupported(
                self.func,
                node,
                "Arrays have no attribute '{}'".format(name),
                hint="Supported: {}.".format(", ".join(sorted(emit.ARRAY_METHODS)) + ", size"),
            )
        if value.type.is_fund:
            raise_unsupported(self.func, node, "'{}' has no attribute '{}'".format(value.type, name))
        # Opaque C++ object: pass the access straight through.
        return _CppAttr(value, name)

    def ex_Call(self, node):
        fail = self.fail_at(node)
        emit.check_no_kwargs(node.keywords, "a translated function", fail)
        # An unshadowed builtin name is dispatched by name; its arguments are
        # translated first, so that an unsupported argument reports itself.
        if (
            isinstance(node.func, ast.Name)
            and self.scope.lookup(node.func.id) is None
            and node.func.id not in self._get_globals()
        ):
            return self._builtin_call(node.func.id, [self.expr(a) for a in node.args], node)

        callee = self._maybe_ref(node.func)

        # np.array([...]) builds an array out of element expressions, and
        # astype() takes a dtype rather than a value; both need their arguments
        # before the ordinary translation.
        if isinstance(callee, _FuncRef) and callee.name == "array":
            return self._array_call(node, fail)
        if isinstance(callee, _BoundMethod) and callee.name == "astype":
            if len(node.args) != 1:
                raise_unsupported(self.func, node, "astype() takes exactly one argument")
            return emit.astype(callee.obj, self._static_value(node.args[0], "astype()"), fail)

        args = [self.expr(a) for a in node.args]

        if isinstance(callee, _BoundMethod):
            return self._array_method_call(callee, args, node)
        if isinstance(callee, _CppAttr):
            return emit.cpp_method(callee.obj, callee.name, args)
        if isinstance(callee, _CppRef):
            return emit.call(callee.cpp_name, args, UNKNOWN_T)
        if isinstance(callee, _DeclaredRef):
            return emit.call(callee.cpp_name, args, callee.return_type or UNKNOWN_T)
        if isinstance(callee, _FuncRef):
            return self._math_call(callee, args, node)
        if isinstance(callee, _ModuleRef):
            raise_unsupported(self.func, node, "'{}' is a module and cannot be called".format(callee.path))

        # A plain name: builtins.
        if isinstance(node.func, ast.Name):
            return self._builtin_call(node.func.id, args, node)
        raise_unsupported(self.func, node, "This call cannot be translated")

    def _static_value(self, node, what):
        """Evaluate an argument that must be known at declaration time."""
        if not self._is_static(node) or not self._fold_safe(node):
            raise_unsupported(
                self.func,
                node,
                "The argument of {} has to be known when the function is declared".format(what),
            )
        try:
            return eval(compile(ast.Expression(body=node), "<pydeclare>", "eval"), dict(self._get_globals()))
        except Exception as exc:
            raise_unsupported(self.func, node, "Cannot evaluate the argument of {}: {}".format(what, exc))

    def _array_call(self, node, fail):
        """Translate np.array([a, b, ...])."""
        if len(node.args) != 1:
            raise_unsupported(self.func, node, "np.array() takes exactly one argument here")
        arg = node.args[0]
        if not isinstance(arg, (ast.List, ast.Tuple)):
            raise_unsupported(
                self.func,
                node,
                "np.array() needs a list or tuple of elements here",
                hint="A constant array is folded automatically; a dynamic one is built from its "
                "elements, as in np.array([x, y]).",
            )
        return emit.array_from_elements([self.expr(e) for e in arg.elts], fail)

    def _array_method_call(self, bound, args, node):
        return emit.array_method(bound.obj, bound.name, args, self.fail_at(node))

    def _math_call(self, ref, args, node):
        fail = self.fail_at(node)
        name = ref.name
        if name in emit.UNARY_FUNCTIONS and len(args) == 1:
            return emit.unary_function(emit.UNARY_FUNCTIONS[name], args[0], fail)
        if name in emit.BINARY_FUNCTIONS and len(args) == 2:
            return emit.binary_function(emit.BINARY_FUNCTIONS[name], args[0], args[1], fail)
        if name in emit.FREE_REDUCTIONS and len(args) == 1:
            if not args[0].type.is_container:
                raise_unsupported(self.func, node, "'{}' expects an array".format(ref.path))
            return emit.array_method(args[0], _REDUCTION_METHOD[emit.FREE_REDUCTIONS[name]], [], fail)
        if name == "where" and len(args) == 3:
            return emit.ternary(args[0], args[1], args[2], fail)
        if name in ("sqrt", "fabs") and len(args) == 1:
            return emit.unary_function(emit.UNARY_FUNCTIONS[name], args[0], fail)
        raise_unsupported(
            self.func,
            node,
            "'{}' is not available in the supported subset (or was called with {} arguments)".format(
                ref.path, len(args)
            ),
        )

    def _builtin_call(self, name, args, node):
        fail = self.fail_at(node)
        if name in BUILTIN_CASTS and len(args) == 1:
            return emit.cast(name, args[0], fail)
        if name == "len" and len(args) == 1:
            return emit.length(args[0], fail)
        if name == "abs" and len(args) == 1:
            return emit.unary_function("Abs", args[0], fail)
        if name == "round" and len(args) == 1:
            return emit.unary_function("Round", args[0], fail)
        if name == "pow" and len(args) == 2:
            return emit.binop("**", args[0], args[1], fail)
        if name in ("min", "max"):
            if len(args) == 1:
                if not args[0].type.is_container:
                    raise_unsupported(self.func, node, "{}() of a single value needs an array".format(name))
                return emit.array_method(args[0], name, [], fail)
            if len(args) == 2:
                if args[0].type.is_container or args[1].type.is_container:
                    raise_unsupported(
                        self.func,
                        node,
                        "{}() of two arrays is ambiguous".format(name),
                        hint="Use np.{}imum(a, b) for the element-wise version.".format(name),
                    )
                return emit.binary_function("Maximum" if name == "max" else "Minimum", args[0], args[1], fail)
        if name == "sum" and len(args) == 1:
            if not args[0].type.is_container:
                raise_unsupported(self.func, node, "sum() expects an array")
            return emit.array_method(args[0], "sum", [], fail)
        if name == "range":
            raise_unsupported(self.func, node, "range() can only be used in a for loop")
        raise_unsupported(
            self.func,
            node,
            "The function '{}' is not available in the supported subset".format(name),
            hint="Supported builtins: abs, bool, float, int, len, max, min, pow, round, sum.",
        )

    def ex_List(self, node):
        raise_unsupported(
            self.func,
            node,
            "List literals are only supported when they do not depend on the arguments",
            hint="A constant list is folded into an RVec literal; a list built from arguments "
            "would need a dynamic container.",
        )

    ex_Tuple = ex_List
    ex_Dict = ex_List
    ex_Set = ex_List

    def ex_ListComp(self, node):
        raise_unsupported(
            self.func,
            node,
            "List comprehensions are not supported",
            hint="Use element-wise array expressions, or an explicit for loop.",
        )

    ex_SetComp = ex_ListComp
    ex_DictComp = ex_ListComp
    ex_GeneratorExp = ex_ListComp

    def ex_Lambda(self, node):
        raise_unsupported(self.func, node, "Nested lambdas are not supported")

    def ex_JoinedStr(self, node):
        raise_unsupported(self.func, node, "f-strings are not supported")

    def ex_Starred(self, node):
        raise_unsupported(self.func, node, "Argument unpacking is not supported")

    # -----------------------------------------------------------------------
    # Names from the enclosing scope
    # -----------------------------------------------------------------------

    def _maybe_ref(self, node):
        """Like expr(), but may also return a reference object."""
        if isinstance(node, ast.Name):
            local = self.scope.lookup(node.id)
            if local is not None:
                return local
            folded = self._try_fold(node)
            if folded is not None:
                return folded
            return self._resolve_global(node.id, node)
        if isinstance(node, ast.Attribute):
            return self.ex_Attribute(node)
        return self.expr(node)

    def _get_globals(self):
        if self.globals_cache is None:
            env = {}
            env.update(getattr(self.func, "__globals__", {}) or {})
            closure = getattr(self.func, "__closure__", None)
            code = getattr(self.func, "__code__", None)
            if closure and code:
                for name, cell in zip(code.co_freevars, closure):
                    try:
                        env[name] = cell.cell_contents
                    except ValueError:
                        pass
            self.globals_cache = env
        return self.globals_cache

    def _resolve_global(self, name, node):
        env = self._get_globals()
        if name not in env:
            import builtins

            if hasattr(builtins, name):
                raise_unsupported(
                    self.func,
                    node,
                    "The builtin '{}' is not available in the supported subset".format(name),
                )
            raise_unsupported(self.func, node, "Undefined name '{}'".format(name))
        return self._classify(env[name], name, node)

    def _classify(self, obj, path, node):
        value = emit.constant_value(obj, self.fail_at(node))
        if value is not None:
            return value

        module_name = getattr(obj, "__name__", None)
        if inspect.ismodule(obj):
            if module_name in ("numpy", "math"):
                return _ModuleRef("numpy" if module_name == "numpy" else "math", path)
            raise_unsupported(
                self.func,
                node,
                "Module '{}' cannot be used in translated code".format(module_name),
                hint="Only numpy and math are understood.",
            )

        declared = getattr(obj, "__pydeclare_cpp_name__", None)
        if declared is not None:
            return _DeclaredRef(declared, getattr(obj, "__pydeclare_return_type__", None))

        cpp_name = _cppyy_name(obj)
        if cpp_name is not None:
            return _CppRef(cpp_name)

        if callable(obj) and module_name in emit.UNARY_FUNCTIONS:
            return _FuncRef(module_name, path)

        raise_unsupported(
            self.func,
            node,
            "Cannot use '{}' (a {}) in translated code".format(path, type(obj).__name__),
            hint="Values captured from the enclosing scope must be numbers, arrays of numbers, "
            "C++ entities known to cling, or other declared callables.",
        )

    def _module_attribute(self, ref, name, node):
        if name in emit.CONSTANTS:
            code, type_ = emit.CONSTANTS[name]
            return emit.Value(code, type_)
        known = (
            name in emit.UNARY_FUNCTIONS
            or name in emit.BINARY_FUNCTIONS
            or name in emit.FREE_REDUCTIONS
            or name in ("where", "array")
        )
        if known:
            return _FuncRef(name, "{}.{}".format(ref.path, name))
        raise_unsupported(
            self.func,
            node,
            "'{}.{}' is not available in the supported subset".format(ref.path, name),
        )

    # -----------------------------------------------------------------------
    # Constant folding
    # -----------------------------------------------------------------------

    def _is_static(self, node):
        """True if the expression only involves names from the enclosing scope."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                if self.scope.lookup(sub.id) is not None:
                    return False
                if sub.id not in self._get_globals():
                    return False
            if isinstance(sub, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                return False
        return True

    def _callee_is_safe(self, func_node):
        """True if this callee is a numpy/math function or a pure builtin."""
        if isinstance(func_node, ast.Name) and func_node.id not in self._get_globals():
            return func_node.id in PURE_BUILTINS
        try:
            code = compile(ast.Expression(body=func_node), "<pydeclare>", "eval")
            obj = eval(code, dict(self._get_globals()))  # noqa: S307
        except Exception:
            return False
        module = (getattr(obj, "__module__", "") or "").split(".")[0]
        if module in ("numpy", "math"):
            return True
        # A bound method of a numpy object, as in np.array([...]).astype(...)
        receiver = getattr(obj, "__self__", None)
        if receiver is not None:
            return (type(receiver).__module__ or "").split(".")[0] == "numpy"
        return False

    def _fold_safe(self, node):
        """True if evaluating this expression at declaration time is harmless.

        Freezing a constant is only sound for pure numerical code.  Evaluating
        an arbitrary call would silently bake in whatever it returned at import
        time, which for something like os.getpid() is plainly wrong, so only
        numpy, math and the pure builtins are allowed to run.
        """
        for child in ast.iter_child_nodes(node):
            if not self._fold_safe(child):
                return False
        if isinstance(node, ast.Call):
            return self._callee_is_safe(node.func)
        return True

    def _try_fold(self, node):
        if isinstance(node, ast.Constant):
            return None  # handled directly, no need to compile anything
        if not self._is_static(node):
            return None
        if not self._fold_safe(node):
            return None
        try:
            code = compile(ast.Expression(body=node), "<pydeclare>", "eval")
            obj = eval(code, dict(self._get_globals()))  # noqa: S307
        except Exception:
            return None
        return emit.constant_value(obj, self.fail_at(node))


class _BoundMethod:
    def __init__(self, obj, name):
        self.obj = obj
        self.name = name


class _CppAttr(emit.Value):
    """Attribute access on an opaque C++ object.

    It is a Value in its own right, so that reading a data member works, and it
    also carries the receiver so that a following call becomes a method call.
    """

    __slots__ = ("obj", "name")

    def __init__(self, obj, name):
        super().__init__("{}.{}".format(obj.paren(emit.PREC_PRIMARY), name), UNKNOWN_T, emit.PREC_PRIMARY)
        self.obj = obj
        self.name = name


_BINOP_NAMES = {
    "Add": "+",
    "Sub": "-",
    "Mult": "*",
    "Div": "/",
    "FloorDiv": "//",
    "Mod": "%",
    "Pow": "**",
    "LShift": "<<",
    "RShift": ">>",
    "BitOr": "|",
    "BitAnd": "&",
    "BitXor": "^",
    "MatMult": "@",
}

_CMP_NAMES = {
    "Eq": "==",
    "NotEq": "!=",
    "Lt": "<",
    "LtE": "<=",
    "Gt": ">",
    "GtE": ">=",
    "Is": "is",
    "IsNot": "is not",
    "In": "in",
    "NotIn": "not in",
}

_REDUCTION_METHOD = {fn: name for name, (fn, _) in emit.ARRAY_METHODS.items()}


def _as_load(node):
    """A copy of an assignment target usable as an expression."""
    clone = ast.parse(ast.unparse(node), mode="eval").body if hasattr(ast, "unparse") else node
    ast.copy_location(clone, node)
    for sub in ast.walk(clone):
        ast.copy_location(sub, node)
    return clone


def _unify(a, b):
    """The type both a and b can be stored in, or unknown."""
    if a == b:
        return a
    if a.is_container != b.is_container:
        return UNKNOWN_T
    merged = ct.promote(a, b)
    return merged


def _cppyy_name(obj):
    """The C++ name of a cppyy class, namespace or function, if it is one."""
    name = getattr(obj, "__cpp_name__", None)
    if isinstance(name, str) and name:
        return name
    module = getattr(obj, "__module__", "") or ""
    qual = getattr(obj, "__name__", None)
    if isinstance(qual, str) and qual and (module.startswith("cppyy") or module == "ROOT"):
        prefix = module.split(".", 1)[1] if module.startswith("cppyy.gbl.") else ""
        return "{}::{}".format(prefix, qual) if prefix else qual
    return None
