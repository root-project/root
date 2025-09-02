"""
Remove all packages present in the root prefix.dev channel https://prefix.dev/channels/root
"""
import json
import requests
import urllib
import os


def main():
    for arch in ["linux-64", "noarch"]:
        with urllib.request.urlopen(f"https://repo.prefix.dev/root/{arch}/repodata.json") as response:
            repodata = response.read().decode()
            jsondata = json.loads(repodata)
            for pkg in jsondata["packages.conda"]:

                token = os.environ["ROOT_PREFIX_CHANNEL_API_TOKEN"]

                headers = {"Authorization": f"Bearer {token}"}

                delete_response = requests.delete(
                    f"https://prefix.dev/api/v1/delete/root/{arch}/{pkg}", headers=headers)
                if delete_response.ok:
                    print(f"Deleted package {delete_response.url}")
                else:
                    print("The DELETE request returned false")


if __name__ == "__main__":
    raise SystemExit(main())
