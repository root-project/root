Debugging in JIT Compiled Code
------------------------------

C++ Language Interoperability Layer - Debugging Guide
======================================================

Overview
========

This guide provides comprehensive instructions for debugging CppInterOp applications using LLDB.

Prerequisites
=============

Before proceeding with debugging, ensure you have the following tools installed:

- **LLDB**: The LLVM debugger
- **LLVM/Clang**: The C++ compiler and toolchain
- **CppInterOp**: The C++ language interoperability library

The debugging tools should be available in your LLVM toolchain. On most systems, these are installed alongside your LLVM/Clang installation.

Setting Up Debug Environment
============================

To effectively debug CppInterOp applications, you need to compile your code with debugging symbols and disable optimizations. This ensures that the debugger can accurately map between source code and machine instructions.

**Compilation Flags**

When compiling your CppInterOp application, use the following essential flags:

.. code-block:: bash

   $CXX -I$CPPINTEROP_DIR/include -g -O0 -lclangCppInterOp -Wl,-rpath,$CPPINTEROP_DIR/build/lib

**Flag Explanation:**

- ``-g``: Includes debugging information in the executable
- ``-O0``: Disables compiler optimizations for clearer debugging
- ``-I$CPPINTEROP_DIR/include``: Includes CppInterOp headers
- ``-lclangCppInterOp``: Links against the CppInterOp library
- ``-Wl,-rpath,$CPPINTEROP_DIR/build/lib``: Sets runtime library path

Creating a Debug-Ready Test Program
===================================

Here's a comprehensive example that demonstrates common CppInterOp usage patterns suitable for debugging:

.. code-block:: cpp

  #include <CppInterOp/CppInterOp.h>
  #include <iostream>

  void run_code(std::string code) {
    Cpp::Declare(code.c_str());
  }

  int main(int argc, char *argv[]) { 
    Cpp::CreateInterpreter({"-gdwarf-4", "-O0"});
    std::vector<Cpp::TCppScope_t> Decls;
    std::string code = R"(
  #include <iostream>
  void f1() {
    std::cout << "in f1 function" << std::endl;
  }
  std::cout << "In codeblock 1" << std::endl;
  int a = 100;
  int b = 1000;
    )";
    run_code(code);
    code = R"(
  f1();
    )";
    run_code(code);
    return 0;
  }


**Program Structure Explanation:**

This example demonstrates key debugging scenarios:

1. **Interpreter Initialization**: The ``Cpp::CreateInterpreter`` call with debug flags
2. **Code Declaration**: Dynamic C++ code execution through ``Cpp::Declare``
3. **Mixed Execution**: Combination of compiled and interpreted code paths
4. **Variable Scoping**: Local variables in both compiled and interpreted contexts

Debugging Strategies
====================

**Debugging Compiled Code**

For debugging the main executable and compiled portions of your CppInterOp application:

.. code-block:: bash

   lldb /path/to/executable
   (lldb) settings set plugin.jit-loader.gdb.enable on
   (lldb) breakpoint set --name f1
   (lldb) r
   1 location added to breakpoint 1
   In codeblock 1
   Process 49132 stopped
   * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
   frame #0: 0x000000010217c008 JIT(0x10215c218) f1() at input_line_1:4:13

**Note**

1. Ensure the JIT loader is enabled to allow LLDB to debug dynamically generated code.
2. Use ``settings set plugin.jit-loader.gdb.enable on`` to enable JIT debugging.
3. Set breakpoints in both compiled and interpreted code using ``breakpoint set --name function_name``.


**Some Caveats**

1. For each block of code, there is a file named ``input_line_<execution_number>`` that contains the code block. This file is in-memory and thus cannot be directly accessed.
2. However, generating actual input_line_<number> files on disk will let LLDB pick them up and render the source content correctly during debugging. This can be achieved by modifying run_code as follows:

.. code-block:: cpp

    void run_code(std::string code) {
        static size_t i = 0;
        i++;
        std::string filename = "input_line_" + std::to_string(i);
        std::ofstream file(filename);
        file << code;
        file.close();
        Cpp::Declare(code.c_str());
    }

.. note::

    You'll need to manually delete these files later to avoid cluttering the filesystem.

3. If a function is called from different cell, then it may take multiple step-ins to reach the function definition due to the way CppInterOp handles code blocks.

Advanced Debugging Techniques
=============================

**Using LLDB with VS Code**

For IDE-based debugging:

1. Install the LLDB extension in VS Code
2. Create a ``launch.json`` configuration:

.. code-block:: json

    {
        "version": "0.2.0",
        "configurations": [
            {
                "type": "lldb-dap",
                "request": "launch",
                "name": "Debug",
                "program": "/path/to/executable",
                "sourcePath" : ["${workspaceFolder}"],
                "cwd": "${workspaceFolder}",
                "initCommands": [
                    "settings set plugin.jit-loader.gdb.enable on", // This is crucial 
                ]
            },
        ]
    }



Further Reading
===============

- **LLDB Documentation**: `LLDB Debugger <https://lldb.llvm.org/>`_
- **CppInterOp Source**: `CppInterOp Repository <https://github.com/compiler-research/CppInterOp>`_
- **Clang Documentation**: `Clang Compiler <https://clang.llvm.org/docs/>`_
- **LLVM Debugging Guide**: `LLVM Debug Info <https://llvm.org/docs/SourceLevelDebugging.html>`_


