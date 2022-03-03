# Authors:
# * Jonas Rembser 01/2022
# * Carsten D. Burgard 01/2022

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


class RooJSONFactoryWSTool(object):
    @classmethod
    def gendoc(cls):
        """Generate the importer and exporter documentation."""
        hs3 = {}

        for key, importer in cls.importers():
            if not key in hs3.keys():
                hs3[key] = {}
            if not "import" in hs3[key]:
                hs3[key]["import"] = []
            hs3[key]["import"].append({"native": True})
        for tclass, exporters in cls.exporters():
            for exp in exporters:
                key = exp.key()
                if not key in hs3.keys():
                    hs3[key] = {}
                hs3[key]["class"] = str(tclass.GetName())
                if not "export" in hs3[key]:
                    hs3[key]["export"] = []
                hs3[key]["export"].append({"native": True})
        for key, importer in cls.pdfImportExpressions():
            if not key in hs3.keys():
                hs3[key] = {}
            if not "import" in hs3[key]:
                hs3[key]["import"] = []
            hs3[key]["import"].append(
                {
                    "class": str(importer.tclass.GetName()),
                    "args": [str(e) for e in importer.arguments],
                    "native": False,
                }
            )
        for key, importer in cls.functionImportExpressions():
            if not key in hs3.keys():
                hs3[key] = {}
            if not "import" in hs3[key]:
                hs3[key]["import"] = []
            hs3[key]["import"].append(
                {
                    "class": str(importer.tclass.GetName()),
                    "args": [str(e) for e in importer.arguments],
                    "native": False,
                }
            )
        for tclass, exporter in cls.exportKeys():
            key = exporter.type
            if not key in hs3.keys():
                hs3[key] = {}
            hs3[key]["class"] = str(tclass.GetName())
            if not "export" in hs3[key]:
                hs3[key]["export"] = []
            hs3[key]["export"].append(
                {
                    "native": False,
                    "proxies": {str(a): str(b) for a, b in exporter.proxies},
                }
            )
        return hs3

    @classmethod
    def writedoc(cls, fname):
        """Write the importer and exporter documentation as LaTeX code."""
        hs3 = cls.gendoc()

        with open(fname, "wt") as outfile:
            outfile.write("\\documentclass{article}\n")
            outfile.write("\\begin{document}\n")
            outfile.write("\\begin{description}\n")
            for key, info in hs3.items():
                outfile.write("\\item[" + key + "]\n")
                if "class" in info.keys():
                    outfile.write("\\texttt{" + info["class"] + "}\n")
                else:
                    outfile.write("~\n")
                outfile.write("\\begin{description}\n")
                if "import" in info.keys():
                    for importer in info["import"]:
                        if importer["native"]:
                            outfile.write("\\item[import] \\textit{(native code)}\n")
                        else:
                            outfile.write(
                                "\\item[import] "
                                + "\\texttt{"
                                + importer["class"]
                                + "("
                                + ",".join(importer["args"])
                                + ")}"
                                + "\n"
                            )
                if "export" in info.keys():
                    for exporter in info["export"]:
                        if exporter["native"]:
                            outfile.write("\\item[export] \\textit{(native code)}\n")
                        else:
                            outfile.write(
                                "\\item[export] "
                                + ", ".join([a + "$\\to$" + b for a, b in exporter["proxies"].items()])
                                + "\n"
                            )
                outfile.write("\\end{description}\n")
            outfile.write("\\end{description}")
            outfile.write("\\end{document}")
