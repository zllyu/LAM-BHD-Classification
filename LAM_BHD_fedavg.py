"""Compatibility import for older references to this module.

NVFlare 2.7 provides the FedAvg workflow, so this project no longer carries a
custom FedAvg implementation.
"""

from nvflare.app_common.workflows.fedavg import FedAvg

__all__ = ["FedAvg"]
