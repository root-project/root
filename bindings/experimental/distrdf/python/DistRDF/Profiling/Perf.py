from typing import Optional
import os

from subprocess import Popen, DEVNULL, PIPE

from DistRDF.Profiling.ErrorHandling import check_and_raise_errors

PERF_DEFAULTS = {
    "-e":"cycles:u",
    "-m":"16K",
    "--max-size":"50M",
    "-F":"99",
}

def remove_logs(stderr:str)->str:
    
    err_lines = stderr.splitlines()
    msg = "perf record:"
    return "\n".join([line for line in err_lines if not msg in line])

class collect_perf_data:
    """
    Context manager to collect profiling data in the mapper.
    Data is saved in proc-pid.perf.data
    """

    def __init__(self, data_dir:str, perf_options:Optional[dict]):

        #get current process id
        pid = os.getpid()

        #build perf options
        os.makedirs(data_dir, exist_ok=True)
        data_name = os.path.join(data_dir,f"proc-{pid}.perf.data")

        customizable_options = PERF_DEFAULTS.copy()
        customizable_options.update(perf_options or {})
        
        self.perf_opts = ["-p",str(pid),"--call-graph=fp","-o",data_name] + list(sum(customizable_options.items(), ()))


    def __enter__(self):
        
        #spaw process hosting perf
        self.perf_proc = Popen(["perf","record"]+self.perf_opts, stdout=DEVNULL, stderr=PIPE)

    def __exit__(self, *exc):

        #terminate perf sampling
        self.perf_proc.terminate()

        #ensure perf has dumped data by waiting for its subprocess to end
        #this is necessary since terminate is a non-blocking operation
        self.perf_proc.wait()

        # Catch possible perf errors
        _, err = self.perf_proc.communicate()
        err = remove_logs(err.decode("utf-8"))
        check_and_raise_errors(err,"perf record")