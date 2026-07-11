from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Dict
from enum import Enum

class CollectionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    EMPTY = "empty"
    RATE_LIMITED = "rate_limited"
    AUTH_FAILED = "auth_failed"
    NETWORK_ERROR = "network_error"
    INVALID_SCHEMA = "invalid_schema"

@dataclass
class CollectionResult:
    """Standard result object for all collectors."""
    source: str
    collected_at: datetime = field(default_factory=datetime.now)
    status: CollectionStatus = CollectionStatus.SUCCESS
    raw_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        return self.status == CollectionStatus.SUCCESS
    
    @property
    def record_count(self) -> int:
        if self.raw_data and isinstance(self.raw_data, (list, dict)):
            return len(self.raw_data)
        return 0
