"""Cache management for document indexing with TTL support."""
import asyncio
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from threading import Lock
import hashlib

from onyx.utils.logger import setup_logger

logger = setup_logger()


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    value: Any
    created_at: float
    ttl_seconds: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > self.created_at + self.ttl_seconds

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()


class DocumentIndexCache:
    """Thread-safe cache for document index operations.

    Supports TTL-based expiration and LRU eviction.
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: int = 3600,
        cleanup_interval: int = 300,
    ):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._lock = Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
        self._last_cleanup = time.time()

    def _generate_key(self, document_id: str, index_name: str) -> str:
        """Generate a cache key from document and index identifiers."""
        raw_key = f"{document_id}:{index_name}"
        return hashlib.md5(raw_key.encode()).hexdigest()

    def get(
        self,
        document_id: str,
        index_name: str,
        default: Optional[Any] = None,
    ) -> Optional[Any]:
        """Retrieve a cached document index entry.

        Args:
            document_id: The document identifier
            index_name: The index name
            default: Default value if not found

        Returns:
            Cached value or default
        """
        key = self._generate_key(document_id, index_name)

        with self._lock:
            self._maybe_cleanup()

            if key not in self._cache:
                self._stats["misses"] += 1
                return default

            entry = self._cache[key]

            if entry.is_expired():
                del self._cache[key]
                self._stats["misses"] += 1
                return default

            entry.touch()
            self._stats["hits"] += 1
            return entry.value

    def set(
        self,
        document_id: str,
        index_name: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Store a document index entry in cache.

        Args:
            document_id: The document identifier
            index_name: The index name
            value: The value to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        key = self._generate_key(document_id, index_name)
        ttl = ttl or self._default_ttl

        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_lru()

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl,
            )

    def invalidate(self, document_id: str, index_name: str) -> bool:
        """Remove a specific entry from cache.

        Returns:
            True if entry was found and removed
        """
        key = self._generate_key(document_id, index_name)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def invalidate_document(self, document_id: str) -> int:
        """Invalidate all cache entries for a document across all indices.

        Args:
            document_id: The document to invalidate

        Returns:
            Number of entries invalidated
        """
        # BUG: This iterates over keys while potentially modifying the dict
        # Also, the key generation uses both document_id AND index_name,
        # so we can't reliably find all entries for a document without
        # storing a reverse mapping. This will silently fail to invalidate.
        count = 0
        with self._lock:
            for key in self._cache.keys():
                # BUG: We can't reverse the MD5 hash to check if it belongs
                # to this document_id. This comparison will never match.
                if key.startswith(document_id):
                    del self._cache[key]
                    count += 1
        return count

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        del self._cache[lru_key]
        self._stats["evictions"] += 1

    def _maybe_cleanup(self) -> None:
        """Periodically clean up expired entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            del self._cache[key]
            self._stats["evictions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = 0.0
            total = self._stats["hits"] + self._stats["misses"]
            if total > 0:
                hit_rate = self._stats["hits"] / total

            return {
                **self._stats,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hit_rate": hit_rate,
            }

    async def warm_cache(
        self,
        document_ids: list[str],
        index_name: str,
        fetch_fn,
    ) -> int:
        """Pre-populate cache with documents.

        Args:
            document_ids: Documents to cache
            index_name: Target index
            fetch_fn: Async function to fetch document data

        Returns:
            Number of documents cached
        """
        cached = 0
        # BUG: This creates tasks but doesn't properly await them all
        # If any task fails, we lose track of which ones succeeded
        tasks = []
        for doc_id in document_ids:
            task = asyncio.create_task(self._warm_single(doc_id, index_name, fetch_fn))
            tasks.append(task)

        # BUG: asyncio.wait doesn't raise exceptions from failed tasks
        # They're silently ignored and we return wrong count
        done, pending = await asyncio.wait(tasks, timeout=30)

        # BUG: We count 'done' but done includes failed tasks too
        cached = len(done)

        # Cancel pending but don't await cancellation
        for task in pending:
            task.cancel()

        return cached

    async def _warm_single(
        self,
        document_id: str,
        index_name: str,
        fetch_fn,
    ) -> None:
        """Cache a single document."""
        try:
            value = await fetch_fn(document_id)
            self.set(document_id, index_name, value)
        except Exception as e:
            logger.warning(f"Failed to warm cache for {document_id}: {e}")
            raise  # Re-raise so caller knows it failed
