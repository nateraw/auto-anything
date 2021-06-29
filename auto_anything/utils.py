from pathlib import Path
from pkg_resources import parse_requirements, require, DistributionNotFound
from typing import List


class MissingRequirementsError(Exception):
    """Thrown when requirements are missing"""

    def __init__(self, missing: List[str]):
        self.missing = missing
        self.message = '\n\nMissing the following requirements:\n  - ' + '\n  - '.join(self.missing)
        super().__init__(self.message)


def check_requirements(requirements_file_path: str):
    req_path = Path(requirements_file_path)
    assert req_path.exists()
    requirements = parse_requirements(req_path.open())
    missing = []
    for r in requirements:
        req = str(r)
        try:
            require(req)
        except DistributionNotFound as e:
            missing.append(req)
    if len(missing) > 0:
        raise MissingRequirementsError(missing)
