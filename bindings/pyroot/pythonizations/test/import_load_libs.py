import os
import re
import unittest


class ImportLoadLibs(unittest.TestCase):
    """
    Test which libraries are loaded during importing ROOT
    """

    # The whitelist is a list of regex expressions that mark wanted libraries
    # Note that the regex has to result in an exact match with the library name.
    known_libs = [
        # libCore and dependencies
        "libCore",
        "libm",
        "liblz4",
        "libxxhash",
        "liblzma",
        "libzstd",
        "libz",
        "libpthread",
        "libc",
        "libdl",
        "libpcre",
        "libpcre2-8",
        # libCling and dependencies
        "libCling.*",
        "librt",
        "libncurses.*",
        "libtinfo",  # by libncurses (on some older platforms)
        # libTree and dependencies
        "libTree",
        "libThread",
        "libRIO",
        "libNet",
        "libImt",
        "libMathCore",
        "libMultiProc",
        "libssl",
        "libcrypt.*",  # by libssl
        "oqsprovider",  # loaded by libssl on e.g. centos10
        "liboqs",  # used by above
        "libtbb",
        "libtbb_debug",
        "libtbbmalloc",
        "liburing",  # by libRIO if uring option is enabled
        # On centos7 libssl links against kerberos pulling in all dependencies below, removed with libssl1.1.0
        "libgssapi_krb5",
        "libkrb5",
        "libk5crypto",
        "libkrb5support",
        "libselinux",
        "libkeyutils",
        "libcom_err",
        "libresolv",
        # cppyy and Python libraries
        "libcppyy.*",
        "libROOTPythonizations.*",
        "libpython.*",
        "libutil.*",
        ".*cpython.*",
        "_.*",
        ".*module",
        "operator",
        "cStringIO",
        "binascii",
        "libbz2",
        "libexpat",
        "ISO8859-1",
        # System libraries and others
        "libnss_.*",
        "ld.*",
        "libffi",
        "libgcc_s",
        # AddressSanitizer runtime and ROOT configuration
        "libclang_rt.asan-.*",
        "libROOTSanitizerConfig",
    ]

    # Verbose mode of the test
    verbose = False

    def test_import(self):
        """
        Test libraries loaded after importing ROOT
        """
        import ROOT

        libs = str(ROOT.gSystem.GetLibraries())

        if self.verbose:
            print("Initial output from ROOT.gSystem.GetLibraries():\n" + libs)

        # Split paths
        libs = libs.split(" ")

        # Get library name without full path and .so* suffix
        libs = [
            os.path.basename(lib).split(".so")[0]
            for lib in libs
            if not lib.startswith("-l") and not lib.startswith("-L")
        ]

        # Check that the loaded libraries are white listed
        bad_libs = []
        good_libs = []
        matched_re = []
        for lib in libs:
            matched = False
            for known_lib in self.known_libs:
                m = re.match(known_lib, lib)
                if m:
                    if m.group(0) == lib:
                        matched = True
                        good_libs.append(lib)
                        matched_re.append(known_lib)
                        break
            if not matched:
                bad_libs.append(lib)

        if self.verbose:
            print("Found whitelisted libraries after importing ROOT with the shown regex match:")
            for lib, matched_lib in zip(good_libs, matched_re):
                print(" - {} ({})".format(lib, matched_lib))
            import sys

            sys.stdout.flush()

        if bad_libs:
            raise Exception(
                "Found not whitelisted libraries after importing ROOT:"
                + "\n - "
                + "\n - ".join(bad_libs)
                + "\nIf the test fails with a library that is loaded on purpose, please add it to the whitelist."
            )


if __name__ == "__main__":
    unittest.main()
