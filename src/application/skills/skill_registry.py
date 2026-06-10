"""
Skill Registry

Manages loading, indexing, and accessing skills from YAML files and vector store.
Implements dual storage: file system (source of truth) + vector DB (fast matching).
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import yaml
import logging

from src.domain.models.skill import Skill
from src.infrastructure.embeddings.embedding_service import IEmbeddingService
from src.infrastructure.vector_store.chromadb_adapter import ChromaDBAdapter

logger = logging.getLogger(__name__)


class SkillLoadError(Exception):
    """Raised when skill loading fails"""
    pass


@dataclass
class SkillMetadata:
    """Metadata about a skill for quick lookup"""
    skill_id: str
    name: str
    file_path: str
    domain: str
    category: str
    active: bool
    version: str
    created_at: str
    usage_count: int
    success_rate: float
    trigger_embedding_id: Optional[str] = None


class SkillRegistry:
    """
    Registry for managing skills with dual storage.

    Storage:
    - File system: YAML files (source of truth)
    - Vector DB: Embedded triggers for fast semantic matching

    Responsibilities:
    - Load skills from YAML files
    - Index skill triggers in vector store
    - Provide fast skill lookup by ID
    - List skills by domain/category
    - Track skill usage and performance
    - Manage skill lifecycle (activate/deactivate)
    """

    def __init__(
        self,
        skills_dir: str = "data/skills",
        registry_file: str = "data/skills/registry.yaml",
        embedding_service: Optional[IEmbeddingService] = None,
        vector_store: Optional[ChromaDBAdapter] = None
    ):
        """
        Initialize skill registry.

        Args:
            skills_dir: Directory containing skill YAML files
            registry_file: Path to registry index file
            embedding_service: Service for generating embeddings
            vector_store: Vector store for skill trigger embeddings
        """
        self.skills_dir = Path(skills_dir)
        self.registry_file = Path(registry_file)
        self.embedding_service = embedding_service
        self.vector_store = vector_store

        # In-memory caches
        self._skills: Dict[str, Skill] = {}
        self._metadata: Dict[str, SkillMetadata] = {}
        self._registry_data: Optional[Dict[str, Any]] = None

        # Load registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry index file"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    self._registry_data = yaml.safe_load(f)

                # Ensure registry_data is a dict
                if not isinstance(self._registry_data, dict):
                    self._registry_data = {'skills': [], 'last_updated': datetime.now().isoformat()}

                # Load metadata for all skills
                for skill_info in self._registry_data.get('skills', []):
                    metadata = SkillMetadata(
                        skill_id=skill_info['skill_id'],
                        name=skill_info['name'],
                        file_path=skill_info['file_path'],
                        domain=skill_info['domain'],
                        category=skill_info.get('category', ''),  # Optional field with default
                        active=skill_info['active'],
                        version=skill_info['version'],
                        created_at=skill_info['created_at'],
                        usage_count=skill_info.get('usage_count', 0),
                        success_rate=skill_info.get('success_rate', 0.0),
                        trigger_embedding_id=skill_info.get('trigger_embedding_id')
                    )
                    self._metadata[metadata.skill_id] = metadata

                logger.info(
                    f"Loaded registry with {len(self._metadata)} skills from {self.registry_file}"
                )
            else:
                logger.warning(f"Registry file not found: {self.registry_file}")
                self._registry_data = {'skills': [], 'last_updated': datetime.now().isoformat()}

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            raise SkillLoadError(f"Failed to load registry: {e}")

    def _load_skill_from_file(self, file_path: str) -> Skill:
        """
        Load skill from YAML file.

        Args:
            file_path: Path to skill YAML file

        Returns:
            Skill instance

        Raises:
            SkillLoadError: If loading fails
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise SkillLoadError(f"Skill file not found: {file_path}")

            with open(path, 'r') as f:
                skill_data = yaml.safe_load(f)

            # Convert YAML to Skill domain model
            # Note: YAML has different structure than domain model
            # YAML has triggers.semantic_patterns, domain model has triggers as List[str]
            triggers_data = skill_data.get('triggers', {})
            if isinstance(triggers_data, dict):
                # Extract semantic patterns from YAML structure
                triggers = triggers_data.get('semantic_patterns', [])
            else:
                triggers = triggers_data if isinstance(triggers_data, list) else []

            # Convert guardrails from list to dict format
            # YAML has guardrails as list, domain model expects dict
            guardrails_data = skill_data.get('guardrails', [])
            if isinstance(guardrails_data, list):
                # Convert list of guardrail objects to dict indexed by type
                guardrails = {}
                for i, guardrail in enumerate(guardrails_data):
                    if isinstance(guardrail, dict):
                        guardrail_type = guardrail.get('type', f'guardrail_{i}')
                        guardrails[guardrail_type] = guardrail
                    else:
                        guardrails[f'guardrail_{i}'] = guardrail
            else:
                guardrails = guardrails_data if isinstance(guardrails_data, dict) else {}

            # Parse steps from YAML
            # YAML has action_type and tool, domain model expects step_type and tool_name
            from src.domain.models.skill import SkillStep, SkillStepType
            steps = []
            for step_data in skill_data.get('steps', []):
                # Map YAML action_type to SkillStepType
                action_type = step_data.get('action_type', 'tool_call')
                if action_type == 'tool_call':
                    step_type = SkillStepType.TOOL_CALL
                elif action_type == 'decision':
                    step_type = SkillStepType.REASONING
                else:
                    step_type = SkillStepType.TOOL_CALL  # Default

                step = SkillStep(
                    step_id=str(step_data.get('step_id', len(steps) + 1)),
                    step_type=step_type,
                    description=step_data.get('name', 'Unnamed step'),
                    tool_name=step_data.get('tool'),
                    parameters=step_data.get('inputs', {}),
                    requires_approval=step_data.get('required', False),
                    approval_reason=step_data.get('approval_reason'),
                    expected_output=', '.join(step_data.get('outputs', []))
                )
                steps.append(step)

            skill = Skill(
                skill_id=skill_data['skill_id'],
                version=skill_data['version'],
                name=skill_data['name'],
                description=skill_data['description'],
                triggers=triggers,
                steps=steps,
                metadata=skill_data.get('metadata', {}),
                guardrails=guardrails
            )

            logger.debug(f"Loaded skill: {skill.skill_id} from {file_path}")
            return skill

        except Exception as e:
            logger.error(f"Failed to load skill from {file_path}: {e}")
            raise SkillLoadError(f"Failed to load skill from {file_path}: {e}")

    def get_skill(self, skill_id: str, force_reload: bool = False) -> Optional[Skill]:
        """
        Get skill by ID from vector DB (no file I/O at runtime).

        Args:
            skill_id: Skill identifier
            force_reload: Force reload from vector DB

        Returns:
            Skill instance or None if not found
        """
        # Check cache first
        if not force_reload and skill_id in self._skills:
            return self._skills[skill_id]

        # Load from vector DB
        if not self.vector_store:
            logger.error("Vector store not configured")
            return None

        try:
            # Search for skill by ID in metadata using ChromaDB operator syntax
            results = self.vector_store.search_by_metadata(
                where={
                    "$and": [
                        {"skill_id": {"$eq": skill_id}},
                        {"doc_type": {"$eq": "skill_trigger"}}
                    ]
                },
                n_results=1
            )

            if not results or "skill_data_json" not in results[0].metadata:
                logger.warning(f"Skill {skill_id} not found in vector DB")
                return None

            # Reconstruct Skill from JSON metadata
            import json
            skill_data = json.loads(results[0].metadata["skill_data_json"])

            from src.domain.models.skill import SkillStep, SkillStepType
            steps = []
            for step_data in skill_data.get("steps", []):
                step = SkillStep(
                    step_id=step_data["step_id"],
                    step_type=SkillStepType(step_data["step_type"]),
                    description=step_data["description"],
                    tool_name=step_data.get("tool_name"),
                    parameters=step_data.get("parameters", {}),
                    requires_approval=step_data.get("requires_approval", False),
                    approval_reason=step_data.get("approval_reason"),
                    expected_output=step_data.get("expected_output")
                )
                steps.append(step)

            skill = Skill(
                skill_id=skill_data["skill_id"],
                version=skill_data["version"],
                name=skill_data["name"],
                description=skill_data["description"],
                triggers=skill_data["triggers"],
                steps=steps,
                metadata=skill_data.get("metadata", {}),
                guardrails=skill_data.get("guardrails", {})
            )

            # Cache it
            self._skills[skill_id] = skill
            logger.debug(f"Loaded skill {skill_id} from vector DB with {len(steps)} steps")
            return skill

        except Exception as e:
            logger.error(f"Failed to load skill {skill_id} from vector DB: {e}")
            return None

    def list_skills(
        self,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        active_only: bool = True
    ) -> List[SkillMetadata]:
        """
        List skills with optional filtering.

        Args:
            domain: Filter by domain
            category: Filter by category
            active_only: Only return active skills

        Returns:
            List of skill metadata
        """
        results = []

        for metadata in self._metadata.values():
            # Apply filters
            if active_only and not metadata.active:
                continue
            if domain and metadata.domain != domain:
                continue
            if category and metadata.category != category:
                continue

            results.append(metadata)

        return results

    def get_all_skills(self, active_only: bool = True) -> List[Skill]:
        """
        Load all skills.

        Args:
            active_only: Only return active skills

        Returns:
            List of Skill instances
        """
        skills = []

        for metadata in self.list_skills(active_only=active_only):
            skill = self.get_skill(metadata.skill_id)
            if skill:
                skills.append(skill)

        return skills

    def index_skill_triggers(self, skill_id: str) -> bool:
        """
        Index skill triggers in vector store with complete skill data.

        Args:
            skill_id: Skill to index

        Returns:
            True if successful
        """
        if not self.embedding_service or not self.vector_store:
            logger.warning("Embedding service or vector store not configured")
            return False

        # Load skill from file first (during indexing only)
        metadata = self._metadata.get(skill_id)
        if not metadata:
            return False

        try:
            skill = self._load_skill_from_file(metadata.file_path)
            if not skill or not skill.triggers:
                logger.warning(f"No triggers for skill {skill_id}")
                return False

            # Enrich trigger text with comprehensive context for better semantic matching
            # This creates a rich embedding that captures:
            # 1. Skill identity (name, description)
            # 2. Trigger patterns (semantic variations)
            # 3. Workflow steps (what the skill actually does)
            # 4. Domain context (booking, billing, amenity, etc.)
            
            # Build enriched text with multiple context layers
            enriched_parts = []
            
            # Layer 1: Core identity
            enriched_parts.append(f"Skill Name: {skill.name}")
            enriched_parts.append(f"Description: {skill.description}")
            
            # Layer 2: Domain and category context
            if skill.metadata:
                domain = skill.metadata.get('domain', '')
                if domain:
                    enriched_parts.append(f"Domain: {domain}")
            
            # Layer 3: Trigger patterns (user intent variations)
            enriched_parts.append(f"User Requests: {', '.join(skill.triggers)}")
            
            # Layer 4: Workflow summary (what the skill does)
            if skill.steps:
                step_descriptions = [step.description for step in skill.steps[:5]]  # First 5 steps
                workflow_summary = ". ".join(step_descriptions)
                enriched_parts.append(f"Workflow: {workflow_summary}")
            
            # Layer 5: Tags for additional context
            if skill.metadata and 'tags' in skill.metadata:
                tags = skill.metadata.get('tags', [])
                if tags:
                    enriched_parts.append(f"Related Topics: {', '.join(tags)}")
            
            # Combine all layers into rich embedding text
            enriched_trigger_text = ". ".join(enriched_parts)
            
            logger.debug(f"Enriched trigger text for {skill_id} ({len(enriched_trigger_text)} chars): {enriched_trigger_text[:200]}...")

            # Generate embedding for enriched text
            embedding_result = self.embedding_service.embed_texts([enriched_trigger_text])
            embedding = embedding_result.embeddings[0]

            # Serialize skill steps for storage
            from src.domain.models.skill import SkillStep
            steps_data = []
            for step in skill.steps:
                steps_data.append({
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "description": step.description,
                    "tool_name": step.tool_name,
                    "parameters": step.parameters,
                    "requires_approval": step.requires_approval,
                    "approval_reason": step.approval_reason,
                    "expected_output": step.expected_output
                })

            # Serialize skill data to JSON string (ChromaDB doesn't support nested dicts)
            import json
            skill_data_json = json.dumps({
                "skill_id": skill.skill_id,
                "version": skill.version,
                "name": skill.name,
                "description": skill.description,
                "triggers": skill.triggers,
                "steps": steps_data,
                "metadata": skill.metadata,
                "guardrails": skill.guardrails
            })

            # Store complete skill data in metadata as JSON string
            # ChromaDB only accepts str, int, float, bool in metadata - no lists or nested dicts
            doc_id = f"skill_{skill_id}"
            self.vector_store.add_documents(
                chunk_ids=[doc_id],
                contents=[enriched_trigger_text],
                embeddings=[embedding],
                metadatas=[{
                    "doc_type": "skill_trigger",
                    "skill_id": skill_id,
                    "domain": str(skill.metadata.get('domain', '')),
                    "category": str(skill.metadata.get('category', '')),
                    "active": True,
                    "skill_name": skill.name,
                    "skill_data_json": skill_data_json  # Store complete skill as JSON string
                }]
            )

            logger.info(f"Indexed skill {skill_id} with {len(steps_data)} steps in vector DB")
            return True

        except Exception as e:
            logger.error(f"Failed to index skill {skill_id}: {e}")
            return False

    def index_all_skills(self) -> int:
        """
        Index all active skills in vector store.

        Returns:
            Number of skills indexed
        """
        count = 0

        for metadata in self.list_skills(active_only=True):
            if self.index_skill_triggers(metadata.skill_id):
                count += 1

        logger.info(f"Indexed {count} skills")
        return count

    def update_skill_metrics(
        self,
        skill_id: str,
        execution_time_ms: Optional[float] = None,
        success: Optional[bool] = None
    ) -> bool:
        """
        Update skill performance metrics.

        Args:
            skill_id: Skill to update
            execution_time_ms: Execution time
            success: Whether execution was successful

        Returns:
            True if successful
        """
        metadata = self._metadata.get(skill_id)
        if not metadata:
            return False

        try:
            # Update in-memory metadata
            if success is not None:
                metadata.usage_count += 1
                if success:
                    # Update success rate (running average)
                    total_successes = metadata.success_rate * (metadata.usage_count - 1)
                    metadata.success_rate = (total_successes + 1) / metadata.usage_count
                else:
                    total_successes = metadata.success_rate * (metadata.usage_count - 1)
                    metadata.success_rate = total_successes / metadata.usage_count

            # Update registry file
            self._save_registry()

            return True

        except Exception as e:
            logger.error(f"Failed to update metrics for {skill_id}: {e}")
            return False

    def _save_registry(self) -> None:
        """Save registry to file"""
        try:
            if not self._registry_data:
                return

            # Update skills list with current metadata
            skills_list = []
            for metadata in self._metadata.values():
                skills_list.append({
                    'skill_id': metadata.skill_id,
                    'file_path': metadata.file_path,
                    'name': metadata.name,
                    'domain': metadata.domain,
                    'category': metadata.category,
                    'active': metadata.active,
                    'version': metadata.version,
                    'created_at': metadata.created_at,
                    'trigger_embedding_id': metadata.trigger_embedding_id,
                    'usage_count': metadata.usage_count,
                    'success_rate': metadata.success_rate
                })

            self._registry_data['skills'] = skills_list
            self._registry_data['last_updated'] = datetime.now().isoformat()

            # Write to file
            with open(self.registry_file, 'w') as f:
                yaml.dump(self._registry_data, f, default_flow_style=False)

            logger.debug(f"Saved registry to {self.registry_file}")

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def activate_skill(self, skill_id: str) -> bool:
        """
        Activate a skill.

        Args:
            skill_id: Skill to activate

        Returns:
            True if successful
        """
        metadata = self._metadata.get(skill_id)
        if not metadata:
            return False

        metadata.active = True
        self._save_registry()
        logger.info(f"Activated skill: {skill_id}")
        return True

    def deactivate_skill(self, skill_id: str) -> bool:
        """
        Deactivate a skill.

        Args:
            skill_id: Skill to deactivate

        Returns:
            True if successful
        """
        metadata = self._metadata.get(skill_id)
        if not metadata:
            return False

        metadata.active = False
        self._save_registry()
        logger.info(f"Deactivated skill: {skill_id}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict with statistics
        """
        active_skills = [m for m in self._metadata.values() if m.active]

        return {
            'total_skills': len(self._metadata),
            'active_skills': len(active_skills),
            'inactive_skills': len(self._metadata) - len(active_skills),
            'total_usage': sum(m.usage_count for m in self._metadata.values()),
            'avg_success_rate': (
                sum(m.success_rate for m in active_skills) / len(active_skills)
                if active_skills else 0.0
            ),
            'domains': list(set(m.domain for m in self._metadata.values())),
            'categories': list(set(m.category for m in self._metadata.values()))
        }
