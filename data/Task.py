from pyhealth.data import  Patient
from torch.utils.data import  Dataset
import sys
import os
sys.path.append(os.path.dirname(__file__)) 
from GraphConstruction import *
from tqdm import *
from pyhealth.medcode import CrossMap
import pandas as pd
from collections import defaultdict
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

mapping = CrossMap("ICD10CM", "CCSCM")
mapping3 = CrossMap("ICD9CM", "CCSCM")
# mapping = CrossMap("SNOMEDCT", "CCSCM")  # Not ICD10CMs
def diag_prediction_mimic4_fn(patient: Patient):
    samples = []
    visit_ls = list(patient.visits.keys())
    for i in range(len(visit_ls)):
        visit = patient.visits[visit_ls[i]]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        cond_ccs = []
        for con in conditions:
            if mapping.map(con):
                cond_ccs.append(mapping.map(con)[0]) 

        if len(cond_ccs) * len(procedures) * len(drugs) == 0:
            continue
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": cond_ccs,
                "procedures": procedures,
                "adm_time" : visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
                "drugs": drugs,
                "cond_hist": conditions,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["cond_hist"] = [samples[0]["cond_hist"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]

    for i in range(1, len(samples)):
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["cond_hist"] = samples[i - 1]["cond_hist"] + [
            samples[i]["cond_hist"]
        ]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [
            samples[i]["adm_time"]
        ]

    for i in range(len(samples)):
        samples[i]["cond_hist"][i] = []

    return samples


def diag_prediction_mimic3_fn(patient: Patient):
    samples = []
    visit_ls = list(patient.visits.keys())
    for i in range(len(visit_ls)):
        visit = patient.visits[visit_ls[i]]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        cond_ccs = []
        for con in conditions:
            if mapping3.map(con):
                cond_ccs.append(mapping3.map(con)[0]) 
        if len(cond_ccs) * len(procedures) * len(drugs) == 0:
            continue
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": cond_ccs,
                "procedures": procedures,
                "adm_time" : visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
                "drugs": drugs,
                "cond_hist": conditions,
            }
        )
    if len(samples) < 2:
        return []
    # add history
    samples[0]["cond_hist"] = [samples[0]["cond_hist"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]

    for i in range(1, len(samples)):
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["cond_hist"] = samples[i - 1]["cond_hist"] + [
            samples[i]["cond_hist"]
        ]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [
            samples[i]["adm_time"]
        ]
    for i in range(len(samples)):
        samples[i]["cond_hist"][i] = []
    return samples

import pandas as pd
import os
from collections import defaultdict
from datetime import datetime
from torch.utils.data import Dataset

import pandas as pd
import os
from collections import defaultdict
from torch.utils.data import Dataset

def diag_prediction_synthea_fn(synthea_records):
    patient_samples = []
    # Dummy mapping class - replace with real implementation
  

    # Create dictionaries for fast lookup
    condition_dict = defaultdict(list)
    for record in synthea_records['conditions']:
        condition_dict[record['ENCOUNTER']].append(record['CODE'])
    
    procedure_dict = defaultdict(list)
    for record in synthea_records['procedures']:
        procedure_dict[record['ENCOUNTER']].append(record['CODE'])
    
    medication_dict = defaultdict(list)
    for record in synthea_records['medications']:
        medication_dict[record['ENCOUNTER']].append(record['CODE'])
    
    # Create patient encounter timeline
    patient_visits = defaultdict(list)
    for record in synthea_records['encounters']:
        patient_visits[record['PATIENT']].append({
            'encounter_id': record['ID'],
            'date': record['DATE'],
            'patient_id': record['PATIENT']  # Ensure patient_id is included
        })
    
    # Process each patient
    for patient_id, visits in patient_visits.items():
        # Sort visits by date
        visits.sort(key=lambda x: x['date'])
        
        samples = []
        for visit in visits:
            enc_id = visit['encounter_id']
            conditions = condition_dict.get(enc_id, [])
            procedures = procedure_dict.get(enc_id, [])
            drugs = medication_dict.get(enc_id, [])
            
            # Skip visits with missing any data type
            if not conditions or not procedures or not drugs:
                continue
                
            # Map conditions to CCS
            cond_ccs = []
            for con in conditions:
                mapped = mapping.map(str(con))
                if mapped:
                    cond_ccs.append(mapped[0])
            
            # Process drug codes (first 4 characters)
            drugs = [str(drug)[:4] for drug in drugs]
            
            samples.append({
                "visit_id": enc_id,
                "patient_id": patient_id,
                "conditions": cond_ccs,
                "procedures": procedures,
                "adm_time": visit['date'],
                "drugs": drugs,
                "cond_hist": conditions  # Raw condition codes
            })
        
        # Skip patients with less than 2 visits
        if len(samples) < 2:
            continue
            
        # Build history sequences (same as MIMIC3)
        samples[0]["cond_hist"] = [samples[0]["cond_hist"]]
        samples[0]["procedures"] = [samples[0]["procedures"]]
        samples[0]["drugs"] = [samples[0]["drugs"]]
        samples[0]["adm_time"] = [samples[0]["adm_time"]]

        for i in range(1, len(samples)):
            samples[i]["drugs"] = samples[i-1]["drugs"] + [samples[i]["drugs"]]
            samples[i]["procedures"] = samples[i-1]["procedures"] + [samples[i]["procedures"]]
            samples[i]["cond_hist"] = samples[i-1]["cond_hist"] + [samples[i]["cond_hist"]]
            samples[i]["adm_time"] = samples[i-1]["adm_time"] + [samples[i]["adm_time"]]
        
        # Mask current visit in history
        for i in range(len(samples)):
            samples[i]["cond_hist"][i] = []
        
        patient_samples.extend(samples)
    print('samples',patient_samples)
#     print(f"Created {len(patient_samples)} samples")
    
    return patient_samples

    
class MMDataset(Dataset):
    def __init__(self, dataset, tokenizer, dim, device, trans_dim=0, di=False):
        self.sequence_dataset = dataset.samples
        self.tokenizer = tokenizer
        self.trans_dim = trans_dim
        self.di = di
        self.dim = dim
        self.device = device
        self.graph_data = PatientGraph(self.tokenizer, self.sequence_dataset, dim=self.dim, device = self.device, trans_dim=self.trans_dim, di=self.di)#.all_data

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        sequence_data = self.sequence_dataset[idx]
        graph_data = self.graph_data[idx]
        return sequence_data, graph_data

