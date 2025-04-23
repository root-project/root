import sys
def main():
    args = sys.argv[1:]
    refargs = ["foo", "bar"]
    if args != refargs:
        raise ValueError(
            "FAILURE: the arguments were %s, while %s were expected." %(args,refargs))
if __name__ == "__main__":
    raise SystemExit(main())