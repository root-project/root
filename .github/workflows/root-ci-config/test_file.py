import os
from build_utils import (
    cmake_options_from_dict,
    die,
    download_latest,
    github_log_group,
    print_info,
    load_config,
    print_shell_log,
    subprocess_with_log,
    upload_file,
    print_warning,
)

WORKDIR = '/tmp/workspace'


def main():
    shell_log = ""
    contents = create_coverage_xml(shell_log)
    print(contents)
    


# @github_log_group("Run tests")
# def run_ctest(shell_log: str, extra_ctest_flags: str) -> str:
#     result, shell_log = subprocess_with_log(f"""
#         cd '{WORKDIR}/build'
#         ctest --output-on-failure --parallel {os.cpu_count()} --output-junit TestResults.xml {extra_ctest_flags}
#     """, shell_log)

#     if result != 0:
#         die(result, "Some tests failed", shell_log)

#     return shell_log

# @github_log_group("Create Test Coverage")
# def create_coverage_html(shell_log: str) -> str:
#     #directory = f"{WORKDIR}/build/interpreter/llvm-project/llvm/lib/ProfileData/Coverage/CMakeFiles"
#     directory = f"../root-ci-config/"
#     result, shell_log = subprocess_with_log(f"""
#         cd '{directory}'
#         lcov --directory . --capture --output-file coverage.info
#         genhtml coverage.info --output-directory coverage_report
#         cd coverage_report
#         firefox index.html
#     """, shell_log)
#     if directory == "":
#         print("No content")
#     #contents = os.listdir(directory)
#     print(result)
#     print("---------")
#     print(shell_log)
#     #return contents


@github_log_group("Create Test Coverage in XML")
def create_coverage_xml(shell_log: str) -> str:
    directory = f"{WORKDIR}/build/interpreter/llvm-project/llvm/lib/ProfileData/Coverage"
    #directory = f"../root-ci-config"
    result, shell_log = subprocess_with_log(f"""
        cd '{directory}'
        pwd
        gcovr --cobertura-pretty -r ~/{WORKDIR}/src . -o XMLCoverage.xml
        pwd
    """, shell_log)
    if directory == "":
        print("No content")
    #contents = os.listdir(directory)
    print(result)
    print("---------")
    print(shell_log)


if __name__ == "__main__":
    main()