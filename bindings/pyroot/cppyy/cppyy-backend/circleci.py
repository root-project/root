import http.client
from typing import Sequence
import argparse
import os
import urllib.request
from pathlib import Path
import time
import json

import sys


def get_artifact(
    token: str,
    vcs: str = "github",
    org: str = "wlav",
    project: str = "cppyy-backend",
    job_number: int = 0,
    **kwargs,
) -> int:

    conn = http.client.HTTPSConnection("circleci.com")

    headers = {"Circle-Token": token}

    conn.request(
        "GET",
        f"/api/v2/project/{vcs}/{org}/{project}/{job_number}/artifacts",
        headers=headers,
    )

    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    url = data["items"][0]["url"]
    path = Path(data["items"][0]["path"])
    path.parent.mkdir(exist_ok=True)

    urllib.request.urlretrieve(url, path)
    time.sleep(1.0)
    print(path)
    return 0


def start_job(
    token: str,
    vcs: str = "github",
    org: str = "wlav",
    project: str = "cppyy-backend",
    build_aarch64_wheel: bool = True,
    branch: str = "master",
    **kwargs,
) -> int:
    import http.client

    conn = http.client.HTTPSConnection("circleci.com")

    headers = {
        "content-type": "application/json",
        "Circle-Token": f"{token}",
    }

    # Start pipeline
    payload = {
        "branch": branch,
        "parameters": {"build_aarch64_wheel": build_aarch64_wheel},
    }
    conn.request(
        "POST",
        f"/api/v2/project/{vcs}/{org}/{project}/pipeline",
        json.dumps(payload),
        headers,
    )
    res = conn.getresponse()
    pipeline_data = json.loads(res.read().decode("utf-8"))
    time.sleep(1.0)

    # Get pipeline id
    conn.request(
        "GET",
        f"/api/v2/project/{vcs}/{org}/{project}/pipeline/{pipeline_data['number']}",
        headers=headers,
    )
    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))

    time.sleep(1.0)

    # Get workflow id
    conn.request(
        "GET",
        f"/api/v2/pipeline/{data['id']}/workflow",
        headers=headers,
    )

    res = conn.getresponse()
    workflow_data = json.loads(res.read().decode("utf-8"))

    time.sleep(1.0)

    # Get job id
    conn.request(
        "GET",
        f"/api/v2/workflow/{workflow_data['items'][0]['id']}/job",
        headers=headers,
    )

    res = conn.getresponse()
    job_data = json.loads(res.read().decode("utf-8"))

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--token", default=os.environ.get("CIRCLE_API_TOKEN"))
    parser.add_argument("mode", choices=["artifact", "job"])
    parser.add_argument("--vcs", default="github")
    parser.add_argument("--org", default="wlav")
    parser.add_argument("--project", default="cppyy-backend")
    parser.add_argument("--job-number", default=0)
    parser.add_argument("--build-aarch64-wheel", default=True)

    kwargs = vars(parser.parse_args(argv))
    mode = kwargs.pop("mode")

    if mode == "artifact":
        return get_artifact(**kwargs)

    if mode == "job":
        return start_job(**kwargs)

    print("Invaild arguments")
    print(f"You entered: {kwargs}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
