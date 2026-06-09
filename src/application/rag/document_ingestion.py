"""
Document ingestion pipeline for RAG system.

Handles loading and preprocessing of various document types:
- Markdown policy documents
- JSON data files (bookings, issues, resolutions)
- Text files
- Future: PDF support

Key Features:
- Multi-format support
- Metadata extraction
- Document type detection
- Batch processing
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum
import re


class SourceType(Enum):
    """Source type for document classification."""
    POLICY = "policy"
    BOOKING = "booking"
    ISSUE = "issue"
    RESOLUTION = "resolution"
    RUNBOOK = "runbook"
    CONVERSATION = "conversation"


@dataclass
class Document:
    """Represents an ingested document with metadata."""
    content: str
    document_id: str
    source_type: SourceType
    source_path: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Ensure metadata is initialized."""
        if not self.metadata:
            self.metadata = {}


class DocumentIngestionPipeline:
    """
    Pipeline for ingesting documents from various sources.
    
    Handles:
    - File loading
    - Format detection
    - Content extraction
    - Metadata enrichment
    """
    
    def __init__(self, base_path: str = "data/mock"):
        """
        Initialize the ingestion pipeline.
        
        Args:
            base_path: Base directory for mock data
        """
        self.base_path = Path(base_path)
    
    def ingest_all(self) -> List[Document]:
        """
        Ingest all documents from the mock data directory.
        
        Returns:
            List of ingested documents
        """
        documents = []
        
        # Ingest policy documents
        policy_docs = self.ingest_policies()
        documents.extend(policy_docs)
        
        # Ingest booking data
        booking_docs = self.ingest_bookings()
        documents.extend(booking_docs)
        
        # Ingest issues
        issue_docs = self.ingest_issues()
        documents.extend(issue_docs)
        
        # Ingest resolutions
        resolution_docs = self.ingest_resolutions()
        documents.extend(resolution_docs)
        
        return documents
    
    def ingest_policies(self) -> List[Document]:
        """
        Ingest policy documents from markdown files.
        
        Returns:
            List of policy documents
        """
        policies_dir = self.base_path / "policies"
        if not policies_dir.exists():
            return []
        
        documents = []
        for policy_file in policies_dir.glob("*.md"):
            try:
                content = policy_file.read_text(encoding='utf-8')
                
                # Extract metadata from filename and content
                policy_name = policy_file.stem.replace('_', ' ').title()
                metadata = self._extract_policy_metadata(content, policy_name)
                
                doc = Document(
                    content=content,
                    document_id=f"policy_{policy_file.stem}",
                    source_type=SourceType.POLICY,
                    source_path=str(policy_file),
                    metadata=metadata
                )
                documents.append(doc)
            except Exception as e:
                print(f"Error ingesting policy {policy_file}: {e}")
        
        return documents
    
    def ingest_bookings(self) -> List[Document]:
        """
        Ingest booking records from JSON.
        
        Each booking becomes a separate document for retrieval.
        
        Returns:
            List of booking documents
        """
        bookings_file = self.base_path / "bookings" / "bookings.json"
        if not bookings_file.exists():
            return []
        
        documents = []
        try:
            with open(bookings_file, 'r') as f:
                bookings = json.load(f)
            
            for booking in bookings:
                # Convert booking to searchable text
                content = self._booking_to_text(booking)
                
                doc = Document(
                    content=content,
                    document_id=f"booking_{booking['booking_id']}",
                    source_type=SourceType.BOOKING,
                    source_path=str(bookings_file),
                    metadata={
                        'booking_id': booking['booking_id'],
                        'guest_name': booking['guest']['first_name'] + ' ' + booking['guest']['last_name'],
                        'room_number': booking['room']['room_number'],
                        'status': booking['status'],
                        'check_in_date': booking['check_in_date'],
                        'check_out_date': booking['check_out_date'],
                        'loyalty_tier': booking['guest']['loyalty_tier']
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error ingesting bookings: {e}")
        
        return documents
    
    def ingest_issues(self) -> List[Document]:
        """
        Ingest customer issues from JSON.
        
        Returns:
            List of issue documents
        """
        issues_file = self.base_path / "issues" / "customer_issues.json"
        if not issues_file.exists():
            return []
        
        documents = []
        try:
            with open(issues_file, 'r') as f:
                issues = json.load(f)
            
            for issue in issues:
                # Convert issue to searchable text
                content = self._issue_to_text(issue)
                
                doc = Document(
                    content=content,
                    document_id=f"issue_{issue['issue_id']}",
                    source_type=SourceType.ISSUE,
                    source_path=str(issues_file),
                    metadata={
                        'issue_id': issue['issue_id'],
                        'issue_type': issue['issue_type'],
                        'channel': issue['channel'],
                        'priority': issue['priority'],
                        'status': issue['status'],
                        'booking_id': issue.get('booking_id'),
                        'guest_name': issue.get('guest_name')
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error ingesting issues: {e}")
        
        return documents
    
    def ingest_resolutions(self) -> List[Document]:
        """
        Ingest historical resolutions from JSON.
        
        Returns:
            List of resolution documents
        """
        resolutions_file = self.base_path / "resolutions" / "historical_resolutions.json"
        if not resolutions_file.exists():
            return []
        
        documents = []
        try:
            with open(resolutions_file, 'r') as f:
                resolutions = json.load(f)
            
            for resolution in resolutions:
                # Convert resolution to searchable text
                content = self._resolution_to_text(resolution)
                
                doc = Document(
                    content=content,
                    document_id=f"resolution_{resolution['resolution_id']}",
                    source_type=SourceType.RESOLUTION,
                    source_path=str(resolutions_file),
                    metadata={
                        'resolution_id': resolution['resolution_id'],
                        'issue_type': resolution['issue_type'],
                        'priority': resolution['priority'],
                        'resolution_time_minutes': resolution['resolution_time_minutes'],
                        'guest_satisfaction': resolution['guest_satisfaction'],
                        'approval_required': resolution['approval_required'],
                        'tools_used': resolution['tools_used']
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error ingesting resolutions: {e}")
        
        return documents
    
    def _extract_policy_metadata(self, content: str, policy_name: str) -> Dict[str, Any]:
        """
        Extract metadata from policy document content.
        
        Args:
            content: Policy document content
            policy_name: Name of the policy
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'policy_name': policy_name,
            'source': policy_name,  # Add source field for citation display
            'doc_type': 'policy'
        }
        
        # Extract last updated date
        date_match = re.search(r'\*\*Last updated:\*\*\s*(\w+\s+\d{4})', content)
        if date_match:
            metadata['last_updated'] = date_match.group(1)
        
        # Extract owner
        owner_match = re.search(r'\*\*Owner:\*\*\s*(.+)', content)
        if owner_match:
            metadata['owner'] = owner_match.group(1).strip()
        
        # Extract document ID
        doc_id_match = re.search(r'\*\*Document ID:\*\*\s*(.+)', content)
        if doc_id_match:
            metadata['document_code'] = doc_id_match.group(1).strip()
        
        # Determine policy domain
        if 'cancellation' in policy_name.lower() or 'refund' in policy_name.lower():
            metadata['domain'] = 'billing'
        elif 'upgrade' in policy_name.lower():
            metadata['domain'] = 'room_management'
        elif 'checkout' in policy_name.lower():
            metadata['domain'] = 'operations'
        elif 'accessibility' in policy_name.lower():
            metadata['domain'] = 'accessibility'
        elif 'loyalty' in policy_name.lower():
            metadata['domain'] = 'loyalty'
        elif 'complaint' in policy_name.lower():
            metadata['domain'] = 'guest_services'
        else:
            metadata['domain'] = 'general'
        
        return metadata
    
    def _booking_to_text(self, booking: Dict[str, Any]) -> str:
        """
        Convert booking JSON to searchable text.
        
        Args:
            booking: Booking dictionary
            
        Returns:
            Text representation of booking
        """
        guest = booking['guest']
        room = booking['room']
        
        text_parts = [
            f"Booking ID: {booking['booking_id']}",
            f"Confirmation Number: {booking['confirmation_number']}",
            f"Guest: {guest['first_name']} {guest['last_name']}",
            f"Email: {guest['email']}",
            f"Phone: {guest['phone']}",
            f"Loyalty Tier: {guest['loyalty_tier']}",
            f"Room Number: {room['room_number']}",
            f"Room Type: {room['room_type']}",
            f"Floor: {room['floor']}",
            f"Rate: ${room['rate_per_night']}/night",
            f"Check-in: {booking['check_in_date']}",
            f"Check-out: {booking['check_out_date']}",
            f"Nights: {booking['nights']}",
            f"Total: ${booking['total_amount']}",
            f"Status: {booking['status']}",
            f"Channel: {booking['booking_channel']}"
        ]
        
        if booking.get('special_requests'):
            text_parts.append(f"Special Requests: {booking['special_requests']}")
        
        return "\n".join(text_parts)
    
    def _issue_to_text(self, issue: Dict[str, Any]) -> str:
        """
        Convert issue JSON to searchable text.
        
        Args:
            issue: Issue dictionary
            
        Returns:
            Text representation of issue
        """
        text_parts = [
            f"Issue ID: {issue['issue_id']}",
            f"Type: {issue['issue_type']}",
            f"Channel: {issue['channel']}",
            f"Priority: {issue['priority']}",
            f"Status: {issue['status']}",
            f"Guest: {issue['guest_name']}",
            f"Booking: {issue['booking_id']}",
            f"Description: {issue['description']}"
        ]
        
        if issue.get('metadata'):
            for key, value in issue['metadata'].items():
                text_parts.append(f"{key}: {value}")
        
        return "\n".join(text_parts)
    
    def _resolution_to_text(self, resolution: Dict[str, Any]) -> str:
        """
        Convert resolution JSON to searchable text.
        
        Args:
            resolution: Resolution dictionary
            
        Returns:
            Text representation of resolution
        """
        text_parts = [
            f"Resolution ID: {resolution['resolution_id']}",
            f"Issue Type: {resolution['issue_type']}",
            f"Priority: {resolution['priority']}",
            f"Resolution Time: {resolution['resolution_time_minutes']} minutes",
            f"Guest Satisfaction: {resolution['guest_satisfaction']}",
            f"Approval Required: {resolution['approval_required']}",
            f"Tools Used: {', '.join(resolution['tools_used'])}",
            f"\nOutcome: {resolution['outcome']}",
            f"\nResolution Steps:"
        ]
        
        for step in resolution['steps']:
            text_parts.append(f"  {step['step_number']}. {step['description']}")
        
        if resolution.get('approval_details'):
            approval = resolution['approval_details']
            text_parts.append(f"\nApproval: {approval['approved_by']}")
            text_parts.append(f"Notes: {approval['approval_notes']}")
        
        return "\n".join(text_parts)
    
    def get_ingestion_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Calculate statistics for ingested documents.
        
        Args:
            documents: List of ingested documents
            
        Returns:
            Statistics dictionary
        """
        if not documents:
            return {}
        
        source_counts = {}
        total_chars = 0
        
        for doc in documents:
            source_type = doc.source_type.value
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
            total_chars += len(doc.content)
        
        return {
            'total_documents': len(documents),
            'by_source_type': source_counts,
            'total_characters': total_chars,
            'avg_document_size': total_chars / len(documents) if documents else 0
        }
