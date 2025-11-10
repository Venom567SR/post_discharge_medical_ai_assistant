from typing import Optional
from pathlib import Path
from loguru import logger

from src.config import PATIENTS_DIR, ERROR_PATIENT_NOT_FOUND, ERROR_MULTIPLE_MATCHES
from src.schemas import PatientRecord, PatientLookupResult
from src.utils.io import load_all_patients
from src.logging_setup import log_tool_call


class PatientDatabase:
    """In-memory patient database with lookup capabilities."""
    
    def __init__(self, patients_dir: Path = PATIENTS_DIR):
        """
        Initialize patient database.
        
        Args:
            patients_dir: Directory containing patient JSON files
        """
        self.patients_dir = patients_dir
        self.patients: list[PatientRecord] = []
        self.name_index: dict[str, list[int]] = {}  # normalized name -> patient indices
        self._load_patients()
    
    def _load_patients(self):
        """Load all patient records from JSON files."""
        try:
            patient_dicts = load_all_patients(self.patients_dir)
            self.patients = [PatientRecord(**p) for p in patient_dicts]
            self._build_name_index()
            logger.info(f"Loaded {len(self.patients)} patient records")
        except Exception as e:
            logger.error(f"Failed to load patients: {e}")
            self.patients = []
    
    def _build_name_index(self):
        """Build index for fast name lookup."""
        self.name_index = {}
        for i, patient in enumerate(self.patients):
            # Index by full name (normalized)
            normalized_name = self._normalize_name(patient.name)
            if normalized_name not in self.name_index:
                self.name_index[normalized_name] = []
            self.name_index[normalized_name].append(i)
            
            # Also index by first and last name separately
            parts = normalized_name.split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = parts[-1]
                
                if first_name not in self.name_index:
                    self.name_index[first_name] = []
                self.name_index[first_name].append(i)
                
                if last_name not in self.name_index:
                    self.name_index[last_name] = []
                self.name_index[last_name].append(i)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for matching (lowercase, strip)."""
        return name.lower().strip()
    
    def get_patient_by_name(self, name: str, user_id: str = "unknown") -> PatientLookupResult:
        """
        Look up patient by name.
        
        Args:
            name: Patient name (full or partial)
            user_id: ID of requesting user (for logging)
        
        Returns:
            PatientLookupResult with patient data or error
        """
        normalized_query = self._normalize_name(name)
        
        # Try exact match first
        if normalized_query in self.name_index:
            indices = self.name_index[normalized_query]
            
            # If multiple exact matches, return error
            if len(indices) > 1:
                log_tool_call(
                    user_id=user_id,
                    tool="patient_db",
                    params={"name": name},
                    result="multiple_matches"
                )
                return PatientLookupResult(
                    success=False,
                    error=ERROR_MULTIPLE_MATCHES,
                    error_type="multiple_matches"
                )
            
            # Single exact match
            patient = self.patients[indices[0]]
            log_tool_call(
                user_id=user_id,
                tool="patient_db",
                params={"name": name},
                result=f"found: {patient.name}"
            )
            return PatientLookupResult(
                success=True,
                patient=patient
            )
        
        # Try fuzzy matching (partial name in full name)
        matches = []
        for patient in self.patients:
            normalized_full_name = self._normalize_name(patient.name)
            if normalized_query in normalized_full_name:
                matches.append(patient)
        
        if len(matches) == 0:
            log_tool_call(
                user_id=user_id,
                tool="patient_db",
                params={"name": name},
                result="not_found"
            )
            return PatientLookupResult(
                success=False,
                error=ERROR_PATIENT_NOT_FOUND,
                error_type="not_found"
            )
        
        if len(matches) > 1:
            log_tool_call(
                user_id=user_id,
                tool="patient_db",
                params={"name": name},
                result=f"multiple_matches: {[p.name for p in matches]}"
            )
            return PatientLookupResult(
                success=False,
                error=ERROR_MULTIPLE_MATCHES,
                error_type="multiple_matches"
            )
        
        # Single fuzzy match
        patient = matches[0]
        log_tool_call(
            user_id=user_id,
            tool="patient_db",
            params={"name": name},
            result=f"found (fuzzy): {patient.name}"
        )
        return PatientLookupResult(
            success=True,
            patient=patient
        )
    
    def reload(self):
        """Reload patients from disk (useful if files change)."""
        self._load_patients()
    
    def count(self) -> int:
        """Get number of patients in database."""
        return len(self.patients)
    
    def list_all_names(self) -> list[str]:
        """Get list of all patient names."""
        return [p.name for p in self.patients]


# Global singleton
_patient_db = None

def get_patient_database() -> PatientDatabase:
    """Get or create the global patient database instance."""
    global _patient_db
    if _patient_db is None:
        _patient_db = PatientDatabase()
    return _patient_db


def lookup_patient(name: str, user_id: str = "unknown") -> PatientLookupResult:
    """
    Convenience function for patient lookup.
    
    Args:
        name: Patient name
        user_id: User ID for logging
    
    Returns:
        PatientLookupResult
    """
    db = get_patient_database()
    return db.get_patient_by_name(name, user_id)
