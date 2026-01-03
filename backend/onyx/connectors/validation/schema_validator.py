"""Schema validation for connector configurations."""
import re
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum

from onyx.utils.logger import setup_logger

logger = setup_logger()


class FieldType(Enum):
    """Supported field types for validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    EMAIL = "email"
    URL = "url"
    REGEX = "regex"


@dataclass
class FieldRule:
    """Validation rule for a single field."""
    field_type: FieldType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    nested_schema: Optional["Schema"] = None
    custom_validator: Optional[callable] = None


@dataclass
class Schema:
    """Schema definition for validation."""
    fields: Dict[str, FieldRule] = field(default_factory=dict)
    allow_extra_fields: bool = False
    strict_types: bool = True


class ValidationError(Exception):
    """Raised when validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


class SchemaValidator:
    """Validates data against defined schemas."""

    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    URL_PATTERN = re.compile(
        r'^https?://[^\s/$.?#].[^\s]*$'
    )

    def __init__(self, schema: Schema):
        self.schema = schema
        self._errors: List[str] = []

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data against the schema.

        Args:
            data: Data dictionary to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        self._errors = []

        if not isinstance(data, dict):
            self._errors.append("Data must be a dictionary")
            raise ValidationError(self._errors)

        # Check for required fields
        for field_name, rule in self.schema.fields.items():
            if rule.required and field_name not in data:
                self._errors.append(f"Missing required field: {field_name}")

        # Check for extra fields
        if not self.schema.allow_extra_fields:
            extra = set(data.keys()) - set(self.schema.fields.keys())
            if extra:
                self._errors.append(f"Unexpected fields: {', '.join(extra)}")

        # Validate each field
        for field_name, value in data.items():
            if field_name in self.schema.fields:
                self._validate_field(field_name, value, self.schema.fields[field_name])

        if self._errors:
            raise ValidationError(self._errors)

        return True

    def _validate_field(
        self,
        field_name: str,
        value: Any,
        rule: FieldRule,
    ) -> None:
        """Validate a single field against its rule."""
        # Allow None for optional fields
        if value is None:
            if rule.required:
                self._errors.append(f"{field_name}: Cannot be null")
            return

        # Type validation
        if not self._check_type(value, rule.field_type):
            self._errors.append(
                f"{field_name}: Expected {rule.field_type.value}, got {type(value).__name__}"
            )
            return

        # Length validation for strings and lists
        if rule.field_type in (FieldType.STRING, FieldType.LIST):
            length = len(value)
            if rule.min_length is not None and length < rule.min_length:
                self._errors.append(
                    f"{field_name}: Length {length} below minimum {rule.min_length}"
                )
            if rule.max_length is not None and length > rule.max_length:
                self._errors.append(
                    f"{field_name}: Length {length} exceeds maximum {rule.max_length}"
                )

        # Range validation for numbers
        if rule.field_type in (FieldType.INTEGER, FieldType.FLOAT):
            if rule.min_value is not None and value < rule.min_value:
                self._errors.append(
                    f"{field_name}: Value {value} below minimum {rule.min_value}"
                )
            if rule.max_value is not None and value > rule.max_value:
                self._errors.append(
                    f"{field_name}: Value {value} exceeds maximum {rule.max_value}"
                )

        # Pattern validation
        if rule.pattern and rule.field_type == FieldType.STRING:
            if not re.match(rule.pattern, value):
                self._errors.append(
                    f"{field_name}: Does not match pattern {rule.pattern}"
                )

        # Email validation
        if rule.field_type == FieldType.EMAIL:
            if not self.EMAIL_PATTERN.match(value):
                self._errors.append(f"{field_name}: Invalid email format")

        # URL validation
        if rule.field_type == FieldType.URL:
            if not self.URL_PATTERN.match(value):
                self._errors.append(f"{field_name}: Invalid URL format")

        # Allowed values validation
        if rule.allowed_values is not None:
            # BUG: Case-sensitive comparison even for strings that should be case-insensitive
            # Also doesn't handle the case where value is a list and we want to check each item
            if value not in rule.allowed_values:
                self._errors.append(
                    f"{field_name}: Value '{value}' not in allowed values: {rule.allowed_values}"
                )

        # Nested schema validation
        if rule.nested_schema and rule.field_type == FieldType.DICT:
            nested_validator = SchemaValidator(rule.nested_schema)
            try:
                nested_validator.validate(value)
            except ValidationError as e:
                for error in e.errors:
                    self._errors.append(f"{field_name}.{error}")

        # List item validation with nested schema
        if rule.nested_schema and rule.field_type == FieldType.LIST:
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    nested_validator = SchemaValidator(rule.nested_schema)
                    try:
                        nested_validator.validate(item)
                    except ValidationError as e:
                        for error in e.errors:
                            self._errors.append(f"{field_name}[{i}].{error}")

        # Custom validator
        if rule.custom_validator:
            try:
                result = rule.custom_validator(value)
                if result is not True and result is not None:
                    self._errors.append(f"{field_name}: {result}")
            except Exception as e:
                self._errors.append(f"{field_name}: Custom validation failed: {e}")

    def _check_type(self, value: Any, expected: FieldType) -> bool:
        """Check if value matches expected type."""
        type_map = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: (int, float),  # Allow int for float fields
            FieldType.BOOLEAN: bool,
            FieldType.LIST: list,
            FieldType.DICT: dict,
            FieldType.EMAIL: str,
            FieldType.URL: str,
            FieldType.REGEX: str,
        }

        expected_types = type_map.get(expected)
        if expected_types is None:
            return True

        # BUG: In Python, bool is a subclass of int, so isinstance(True, int) returns True
        # This means boolean values will pass integer validation
        if self.schema.strict_types and expected == FieldType.INTEGER:
            # BUG: We try to fix the bool/int issue but check the wrong thing
            # We should check: if type(value) == bool: return False
            # Instead we check isinstance which doesn't help
            if isinstance(value, bool):
                return True  # BUG: Should return False!

        return isinstance(value, expected_types)


def build_connector_schema(connector_type: str) -> Schema:
    """Build validation schema for a connector type.

    Args:
        connector_type: Type of connector

    Returns:
        Schema for validating connector config
    """
    # Common fields for all connectors
    common_fields = {
        "name": FieldRule(FieldType.STRING, required=True, min_length=1, max_length=100),
        "enabled": FieldRule(FieldType.BOOLEAN, required=False),
        "refresh_interval": FieldRule(FieldType.INTEGER, required=False, min_value=60),
    }

    # Connector-specific schemas
    connector_schemas = {
        "github": {
            **common_fields,
            "access_token": FieldRule(FieldType.STRING, required=True, min_length=10),
            "repositories": FieldRule(FieldType.LIST, required=True, min_length=1),
            "include_issues": FieldRule(FieldType.BOOLEAN, required=False),
            "include_prs": FieldRule(FieldType.BOOLEAN, required=False),
        },
        "slack": {
            **common_fields,
            "bot_token": FieldRule(FieldType.STRING, required=True, pattern=r"^xoxb-"),
            "channels": FieldRule(FieldType.LIST, required=False),
            "workspace_id": FieldRule(FieldType.STRING, required=True),
        },
        "confluence": {
            **common_fields,
            "base_url": FieldRule(FieldType.URL, required=True),
            "username": FieldRule(FieldType.EMAIL, required=True),
            "api_token": FieldRule(FieldType.STRING, required=True),
            "spaces": FieldRule(FieldType.LIST, required=False),
        },
    }

    fields = connector_schemas.get(connector_type, common_fields)
    return Schema(fields=fields, allow_extra_fields=True)


def validate_connector_config(
    connector_type: str,
    config: Dict[str, Any],
) -> List[str]:
    """Validate connector configuration.

    Args:
        connector_type: Type of connector
        config: Configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    schema = build_connector_schema(connector_type)
    validator = SchemaValidator(schema)

    try:
        validator.validate(config)
        return []
    except ValidationError as e:
        return e.errors
