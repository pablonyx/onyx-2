"""
Connector synchronization service.

Handles syncing connector data with external sources while managing
concurrent access and state updates.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class SyncStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SyncJob:
    """Represents a connector sync job."""

    id: UUID
    connector_id: UUID
    status: SyncStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    documents_synced: int = 0
    error_message: Optional[str] = None


@dataclass
class SyncManager:
    """Manages connector synchronization jobs."""

    _jobs: dict[UUID, SyncJob] = field(default_factory=dict)
    _active_syncs: dict[UUID, asyncio.Task] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def start_sync(self, connector_id: UUID) -> SyncJob:
        """Start a new sync job for a connector.

        BUG: Race condition - checks and updates are not atomic.
        Multiple calls can start duplicate syncs for the same connector.
        """
        # Check if sync is already running for this connector
        if connector_id in self._active_syncs:
            task = self._active_syncs[connector_id]
            if not task.done():
                raise ValueError(f"Sync already in progress for connector {connector_id}")

        # Create new job
        job = SyncJob(
            id=uuid4(),
            connector_id=connector_id,
            status=SyncStatus.PENDING,
        )
        self._jobs[job.id] = job

        # Start the sync task
        # BUG: Between the check above and this assignment, another call could
        # have started a sync. This creates duplicate concurrent syncs.
        task = asyncio.create_task(self._run_sync(job))
        self._active_syncs[connector_id] = task

        return job

    async def _run_sync(self, job: SyncJob) -> None:
        """Execute the sync operation."""
        job.status = SyncStatus.IN_PROGRESS
        job.started_at = datetime.now()

        try:
            # Simulate fetching documents from external source
            await asyncio.sleep(0.1)  # Network latency

            # Simulate processing documents
            for i in range(100):
                await asyncio.sleep(0.01)
                job.documents_synced = i + 1

            job.status = SyncStatus.COMPLETED
            job.completed_at = datetime.now()

        except Exception as e:
            job.status = SyncStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()

        finally:
            # Clean up active sync reference
            if job.connector_id in self._active_syncs:
                del self._active_syncs[job.connector_id]

    async def get_job_status(self, job_id: UUID) -> Optional[SyncJob]:
        """Get the status of a sync job."""
        return self._jobs.get(job_id)

    async def cancel_sync(self, connector_id: UUID) -> bool:
        """Cancel an active sync for a connector."""
        if connector_id not in self._active_syncs:
            return False

        task = self._active_syncs[connector_id]
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        return True


# Global sync manager instance
_sync_manager: Optional[SyncManager] = None


def get_sync_manager() -> SyncManager:
    """Get the global sync manager instance."""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = SyncManager()
    return _sync_manager


async def sync_connector(connector_id: UUID) -> SyncJob:
    """Public API to start syncing a connector."""
    manager = get_sync_manager()
    return await manager.start_sync(connector_id)
