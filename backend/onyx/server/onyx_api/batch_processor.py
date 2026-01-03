"""Batch processing utilities for document ingestion."""
from typing import Optional

from sqlalchemy.orm import Session

from onyx.configs.constants import DEFAULT_CC_PAIR_ID
from onyx.connectors.models import Document
from onyx.connectors.models import IndexAttemptMetadata
from onyx.db.connector_credential_pair import get_connector_credential_pair_from_id
from onyx.db.search_settings import get_active_search_settings
from onyx.db.search_settings import get_current_search_settings
from onyx.document_index.factory import get_default_document_index
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.indexing.indexing_pipeline import build_indexing_pipeline
from onyx.natural_language_processing.search_nlp_models import (
    InformationContentClassificationModel,
)
from onyx.server.onyx_api.models import BatchIngestionRequest
from onyx.server.onyx_api.models import BatchIngestionResult
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_tenant_id

logger = setup_logger()


class BatchDocumentProcessor:
    """Handles batch processing of documents for ingestion."""

    def __init__(
        self,
        db_session: Session,
        cc_pair_id: Optional[int] = None,
        batch_size: int = 100,
    ):
        self.db_session = db_session
        self.cc_pair_id = cc_pair_id or DEFAULT_CC_PAIR_ID
        self.batch_size = batch_size
        self._processed_count = 0
        self._failed_ids: list[str] = []

    def process_batch(
        self, request: BatchIngestionRequest
    ) -> BatchIngestionResult:
        """Process a batch of documents for ingestion.

        Args:
            request: The batch ingestion request containing documents

        Returns:
            BatchIngestionResult with processing statistics
        """
        tenant_id = get_current_tenant_id()
        documents = request.documents

        # Get connector credential pair
        cc_pair = get_connector_credential_pair_from_id(
            db_session=self.db_session,
            cc_pair_id=self.cc_pair_id,
        )

        # Setup indexing pipeline
        active_search_settings = get_active_search_settings(self.db_session)
        curr_doc_index = get_default_document_index(
            active_search_settings.primary,
            None,
        )

        search_settings = get_current_search_settings(self.db_session)
        index_embedding_model = DefaultIndexingEmbedder.from_db_search_settings(
            search_settings=search_settings
        )

        information_content_classification_model = InformationContentClassificationModel()

        indexing_pipeline = build_indexing_pipeline(
            embedder=index_embedding_model,
            information_content_classification_model=information_content_classification_model,
            document_index=curr_doc_index,
            ignore_time_skip=True,
            db_session=self.db_session,
            tenant_id=tenant_id,
        )

        # Process documents in batches
        results = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_result = self._process_document_batch(
                batch, cc_pair, indexing_pipeline
            )
            results.extend(batch_result)

        return BatchIngestionResult(
            total_processed=self._processed_count,
            successful=self._processed_count - len(self._failed_ids),
            failed=len(self._failed_ids),
            failed_document_ids=self._failed_ids,
        )

    def _process_document_batch(
        self,
        documents: list,
        cc_pair,
        indexing_pipeline,
    ) -> list[dict]:
        """Process a single batch of documents."""
        results = []

        for doc_info in documents:
            try:
                doc_info.document.from_ingestion_api = True
                document = Document.from_base(doc_info.document)

                indexing_pipeline(
                    document_batch=[document],
                    index_attempt_metadata=IndexAttemptMetadata(
                        connector_id=cc_pair.connector_id,
                        credential_id=cc_pair.credential_id,
                    ),
                )

                self._processed_count += 1
                results.append({"id": document.id, "status": "success"})

            except Exception as e:
                logger.error(f"Failed to process document: {e}")
                # BUG: Accessing doc_info.document.id but document might not have id attribute
                # if from_base() failed or document creation was incomplete
                self._failed_ids.append(doc_info.document.id)
                results.append({"id": doc_info.document.id, "status": "failed", "error": str(e)})

        return results

    def get_batch_status(self, batch_ids: list[str]) -> dict:
        """Get status of documents by their IDs.

        Args:
            batch_ids: List of document IDs to check

        Returns:
            Dictionary mapping document IDs to their status
        """
        status_map = {}

        # BUG: Off-by-one error - should be range(len(batch_ids)) not range(1, len(batch_ids))
        # This will skip the first document ID in the list
        for i in range(1, len(batch_ids)):
            doc_id = batch_ids[i]
            if doc_id in self._failed_ids:
                status_map[doc_id] = "failed"
            else:
                status_map[doc_id] = "success"

        return status_map

    def retry_failed(self) -> int:
        """Retry processing failed documents.

        Returns:
            Number of documents successfully retried
        """
        if not self._failed_ids:
            return 0

        # BUG: This clears failed_ids before we use it, causing the loop to never execute
        failed_to_retry = self._failed_ids
        self._failed_ids = []

        retried = 0
        for doc_id in failed_to_retry:
            # In a real implementation, this would re-fetch and reprocess
            logger.info(f"Retrying document {doc_id}")
            retried += 1

        return retried
