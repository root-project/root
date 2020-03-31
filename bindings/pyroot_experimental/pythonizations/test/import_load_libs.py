import unittest
import re


class ImportLoadLibs(unittest.TestCase):
    """
    Test which libraries are loaded during importing ROOT
    """

    # The whitelist is a list of regex expressions that mark wanted libraries
    # Note that the regex has to result in an exact match with the library name.
    known_libs = [
            # ROOT libraries
            'libCore',
            'libCling',
            'libcppyy.*',
            'libROOTPythonizations.*',
            # ROOT dependencies
            '',
            # System and Python libraries
            'libpthread',
            'libpython.*',
            'math',
            'libm',
            'libdl',
            'libc',
            '_.*',
            'ld.*',
            'select',
            'binascii',
            'grp',
            ]

    # Verbose mode of the test
    verbose = True

    def test_import(self):
        """
        Test libraries loaded after importing ROOT
        """
        import ROOT
        libs = str(ROOT.gSystem.GetLibraries())

        # Split paths
        libs = libs.split(' ')

        # Get library name without full path and .so* suffix
        libs = [l.strip().split('/')[-1] for l in libs]
        libs = [l.strip().split('.')[0] for l in libs]

        # Check that the loaded libraries are white listed
        bad_libs = []
        good_libs = []
        matched_re = []
        for l in libs:
            matched = False
            for r in self.known_libs:
                m = re.search(r, l)
                if m:
                    if m[0] == l:
                        matched = True
                        good_libs.append(l)
                        matched_re.append(r)
                        break
            if not matched:
                bad_libs.append(l)

        if self.verbose:
            print('Found whitelisted libraries after importing ROOT with the shown regex match:')
            for l, r in zip(good_libs, matched_re):
                print(' - {} ({})'.format(l, r))

        if bad_libs:
            raise Exception('Found not whitelisted libraries after importing ROOT:' \
                    + '\n - ' + '\n - '.join(bad_libs) \
                    + '\nIf the test fails with a library that is loaded on purpose, please add it to the whitelist.')
