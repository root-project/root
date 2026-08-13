import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

STRIP_PATTERNS = ("*.pcm", "modules.idx")


def strip_one_wheel(wheel_path: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="strip-wheel-pcms-") as tmp:
        tmp = pathlib.Path(tmp)
        unpack_dir = tmp / "unpacked"
        subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", str(wheel_path), "-d", str(unpack_dir)],
            check=True,
        )

        # `wheel unpack` creates one <name>-<version> subdirectory
        (extracted,) = list(unpack_dir.iterdir())

        removed = []
        for pattern in STRIP_PATTERNS:
            for f in extracted.rglob(pattern):
                f.unlink()
                removed.append(str(f.relative_to(extracted)))

        print(f"{wheel_path.name}: removed {len(removed)} file(s)")

        repacked_dir = tmp / "repacked"
        repacked_dir.mkdir()
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", str(extracted), "-d", str(repacked_dir)],
            check=True,
        )

        (new_wheel,) = list(repacked_dir.glob("*.whl"))
        wheel_path.unlink()
        shutil.move(str(new_wheel), str(wheel_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path, help="directory containing .whl files to strip")
    args = parser.parse_args()

    wheels = sorted(args.directory.glob("*.whl"))
    if not wheels:
        print(f"No .whl files found in {args.directory}", file=sys.stderr)
        return 1

    for wheel in wheels:
        strip_one_wheel(wheel)

    return 0


if __name__ == "__main__":
    sys.exit(main())
