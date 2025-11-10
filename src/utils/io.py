"""Data generation utilities and CLI for creating dummy patient records."""

import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from src.config import PATIENTS_DIR


# Sample data for generating diverse patient records

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Krishna", "Rohan", "Kunal", "Sahil",
    "Priya", "Aditi", "Ananya", "Pooja", "Sneha", "Riya", "Neha", "Kavya"
]

LAST_NAMES = [
    "Sharma", "Patel", "Reddy", "Nair", "Khan", "Singh", "Joshi", "Das",
    "Verma", "Gupta", "Yadav", "Mehta", "Choudhary", "Mishra", "Kulkarni"
]


KIDNEY_DIAGNOSES = [
    "Chronic Kidney Disease Stage 3",
    "Chronic Kidney Disease Stage 4",
    "Acute Kidney Injury",
    "Diabetic Nephropathy",
    "Hypertensive Nephrosclerosis",
    "Glomerulonephritis",
    "Polycystic Kidney Disease",
    "Kidney Stones (Nephrolithiasis)",
    "Urinary Tract Infection with Complications",
    "Post-Transplant Follow-up"
]

SECONDARY_CONDITIONS = [
    "Type 2 Diabetes Mellitus",
    "Hypertension",
    "Coronary Artery Disease",
    "Congestive Heart Failure",
    "Anemia",
    "Hyperparathyroidism",
    "Metabolic Acidosis",
    "Fluid Overload",
    "Electrolyte Imbalance",
    "Peripheral Neuropathy"
]

MEDICATIONS = [
    "Lisinopril 10mg daily",
    "Furosemide 40mg twice daily",
    "Metformin 500mg twice daily",
    "Amlodipine 5mg daily",
    "Atorvastatin 20mg at bedtime",
    "Epoetin alfa injection weekly",
    "Sodium bicarbonate 650mg three times daily",
    "Sevelamer 800mg with meals",
    "Insulin glargine 20 units at bedtime",
    "Aspirin 81mg daily",
    "Calcitriol 0.25mcg daily",
    "Metoprolol 50mg twice daily"
]

PROCEDURES = [
    "Hemodialysis catheter placement",
    "Kidney biopsy",
    "Peritoneal dialysis catheter insertion",
    "Arteriovenous fistula creation",
    "Kidney stone removal (ureteroscopy)",
    "Blood transfusion",
    "Central line placement",
    "Ultrasound-guided kidney drainage"
]

WARNING_SIGNS = [
    "Decreased urine output or dark-colored urine",
    "Severe swelling in legs, ankles, or face",
    "Difficulty breathing or chest pain",
    "Confusion or unusual drowsiness",
    "Severe nausea or vomiting",
    "Blood in urine",
    "Fever above 101°F (38.3°C)",
    "Irregular heartbeat or palpitations",
    "Severe headache or vision changes",
    "Uncontrolled blood pressure readings"
]

FOLLOW_UP_INSTRUCTIONS = [
    "Monitor blood pressure daily and log readings",
    "Weigh yourself daily and report sudden weight gain (>2 lbs in 24 hours)",
    "Limit sodium intake to 2000mg per day",
    "Restrict fluid intake to 1.5 liters per day",
    "Take all medications as prescribed",
    "Schedule lab work for kidney function tests in 2 weeks",
    "Attend dialysis sessions as scheduled (3 times per week)",
    "Follow low-potassium diet",
    "Report any signs of infection immediately",
    "Keep follow-up appointments with nephrologist"
]


def generate_patient_record(patient_id: int) -> dict:
    """Generate a single dummy patient discharge record."""
    
    # Basic info
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    name = f"{first_name} {last_name}"
    
    # Dates
    discharge_date = datetime.now() - timedelta(days=random.randint(1, 90))
    admission_date = discharge_date - timedelta(days=random.randint(2, 14))
    
    # Diagnoses
    primary_diagnosis = random.choice(KIDNEY_DIAGNOSES)
    num_secondary = random.randint(1, 4)
    secondary_diagnoses = random.sample(SECONDARY_CONDITIONS, num_secondary)
    
    # Procedures
    num_procedures = random.randint(0, 3)
    procedures = random.sample(PROCEDURES, num_procedures) if num_procedures > 0 else []
    
    # Medications
    num_meds = random.randint(3, 8)
    medications = random.sample(MEDICATIONS, num_meds)
    
    # Warning signs
    num_warnings = random.randint(3, 6)
    warning_signs = random.sample(WARNING_SIGNS, num_warnings)
    
    # Follow-up instructions
    num_instructions = random.randint(4, 7)
    follow_up_instructions = random.sample(FOLLOW_UP_INSTRUCTIONS, num_instructions)
    
    # Next appointment
    next_appointment_date = discharge_date + timedelta(days=random.randint(7, 30))
    next_appointment = next_appointment_date.strftime("%Y-%m-%d") + " at 2:00 PM"
    
    # Discharge summary
    discharge_summary = f"""Patient {name} was admitted on {admission_date.strftime('%Y-%m-%d')} with {primary_diagnosis}. 
During hospitalization, patient was managed with {', '.join(medications[:3])}. 
{'Procedures performed included ' + ', '.join(procedures) + '.' if procedures else 'No procedures were performed.'}
Patient's condition improved and was stable at discharge. 
Discharged home with instructions to follow up with nephrologist and continue medications as prescribed. 
Patient educated on warning signs and dietary restrictions."""
    
    return {
        "patient_id": f"PT{patient_id:05d}",
        "name": name,
        "discharge_date": discharge_date.strftime("%Y-%m-%d"),
        "admission_date": admission_date.strftime("%Y-%m-%d"),
        "primary_diagnosis": primary_diagnosis,
        "secondary_diagnoses": secondary_diagnoses,
        "procedures": procedures,
        "medications": medications,
        "warning_signs": warning_signs,
        "follow_up_instructions": follow_up_instructions,
        "next_appointment": next_appointment,
        "discharge_summary": discharge_summary
    }


def generate_patients(count: int, output_dir: Path = PATIENTS_DIR) -> list[str]:
    """
    Generate multiple patient records and save to JSON files.
    
    Args:
        count: Number of patients to generate
        output_dir: Directory to save JSON files
    
    Returns:
        List of generated file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    for i in range(1, count + 1):
        patient = generate_patient_record(i)
        filename = f"patient_{patient['patient_id']}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(patient, f, indent=2, ensure_ascii=False)
        
        generated_files.append(str(filepath))
        
    print(f"✓ Generated {count} patient records in {output_dir}")
    return generated_files


def load_all_patients(patients_dir: Path = PATIENTS_DIR) -> list[dict]:
    """
    Load all patient JSON files from directory.
    
    Args:
        patients_dir: Directory containing patient JSON files
    
    Returns:
        List of patient records
    """
    if not patients_dir.exists():
        return []
    
    patients = []
    for json_file in patients_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                patient = json.load(f)
                patients.append(patient)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return patients


def main():
    """CLI entry point for patient data generation."""
    parser = argparse.ArgumentParser(description="Generate dummy patient discharge records")
    parser.add_argument(
        "--generate-patients",
        type=int,
        metavar="N",
        help="Generate N patient records"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PATIENTS_DIR),
        help=f"Output directory (default: {PATIENTS_DIR})"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing patient records"
    )
    
    args = parser.parse_args()
    
    if args.generate_patients:
        output_dir = Path(args.output_dir)
        files = generate_patients(args.generate_patients, output_dir)
        print(f"Generated files: {len(files)}")
        
    elif args.list:
        patients = load_all_patients(Path(args.output_dir))
        print(f"\nFound {len(patients)} patient records:")
        for patient in patients:
            print(f"  - {patient['name']} ({patient['patient_id']}): {patient['primary_diagnosis']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
