import os
import sys
import re

# List of (pattern, replacement) conversions.
general_conversions = [
    ("gSystem", "ROOT.gSystem"),
    ("gROOT", "ROOT.gROOT"),
    ("NULL", "ROOT.kNone"),
    ("kFalse", "False"),
    ("false", "False"),
    ("std::cout << ", "print(f"),
    ("cout << ", "print(f"),
    (" << std::endl", ")"),
    (" << endl", ")"),
    ("void", "def"),
    ("\n\{", ":"),
    ("{", ""),
    ("}", ""),
    ("!", "not "),
    ("\|\|", "or"),
    ("&&", "and"),
    (r"\" << (.*?) << \"", r"{\1}"), # convert cout << to format string
    (r"if \((.*)\)", r"if \1:"),
    ("else", "else:"),
    (r"strcmp\((.*), (.*)\)", r"\1 == \2"),
    (r"std::unique_ptr<(.*?)>", ""),
    (";", ""),
    ("->", "."),
    ("::", "."),
    ("\*", ""),
]


def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
            return i
    return -1


def add_include(includes, module, attr):
    if module == "":
        includes["ROOT"].append(attr)
    else:
        if module not in includes:
            includes[module] = [attr]
        else:
            includes[module].append(attr)


def convert_file(fname, output_file):
    with open(output_file, "w") as out:
        with open(fname, "r") as f:
            lines = f.readlines()
            define_class = False
            includes = {"ROOT": []}
            indent = 0
            for line in lines:
                if "#include" in line:
                    line = line.replace("#include ", "").strip()
                    line = re.sub(r'["<>]', "", line)
                    line = line.replace(".h", "")
                    module = ".".join(line.split("/")[:-1])
                    attr = line.split("/")[-1]
                    add_include(includes, module, attr)
                    continue
                if "using namespace" in line:
                    continue

                # Convert comments
                line = line.replace("///", "##")
                line = line.replace("//", "#")
                line = line.replace("/*", '"""')
                line = line.replace("*/", '"""')

                out.write("   " * indent)

                if not line.strip().startswith("#"):
                    if "class" in line:
                        define_class = True
                        indent = 0
                        out.write("ROOT.gInterpreter.Declare('''\n")

                    indent += sum([t == "{" for t in line])
                    indent -= sum([t == "}" for t in line])

                    if define_class:
                        if indent == 0:
                            out.write("''')\n")
                            define_class = False
                    else:
                        if "=" in line:
                            s = line.split()
                            pos = index_containing_substring(s, "=")
                            line = " ".join(s[pos - 1 :])
                            line = line.replace("new", "")

                        # FIXME: don´t replace these if within quotes
                        for pattern, new in general_conversions:
                            line = re.sub(pattern, new, line)

                out.write(line.strip() + "\n")

    with open(output_file, "r+") as out:
        content = out.read()
        out.seek(0, 0)
        out.write("import ROOT\n\n")
        for k, v in includes.items():
            out.write(f"from {k} import {', '.join(v)}\n")
        out.write(content)


if __name__ == "__main__":
    if len(sys.argv) < 2 or "-h" in sys.argv:
        print("Usage: python3 CtoPyConverter <file_name>")
        exit()

    fname = sys.argv[1]
    output = fname.strip(".C") + ".py"
    convert_file(fname, output)
