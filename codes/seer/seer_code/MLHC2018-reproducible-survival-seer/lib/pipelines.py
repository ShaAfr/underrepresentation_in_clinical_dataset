import lib.filter_column as fc

""" Handles all existing 133 input variables of the SEER ASCII files. Also shows removed entries for completeness. """
data_pipeline_full = [

    # Categorical inputs:
    ('SEER registry', [], [], 'categorical'),
    ('Marital status at diagnosis', [], [], 'categorical'),
    ('Race/ethnicity', [], [], 'categorical'),
    ('NHIA Derived Hisp Origin', [], [], 'categorical'),
    ('Sex', [], [], 'categorical'),
    ('Month of diagnosis', [], [], 'categorical'),
    ('Primary site ICD-O-2 (1973+)', [], [], 'categorical'),
    ('Laterality', [], [], 'categorical'),
    ('Histologic Type ICD-O-2', [], [], 'categorical'),
    ('Behavior Code ICD-O-2', [], [], 'categorical'),
    ('Histologic Type ICD-O-3', [], [], 'categorical'),
    ('Behavior code ICD-O-3', [], [], 'categorical'),
    ('Grade', [], [], 'categorical'),
    ('Type of reporting source', [], [], 'categorical'),
    ('EOD 10 - size (1988+)', [], [], 'categorical'),
    ('EOD 10 - extension', [], [], 'categorical'),
    ('EOD 10 - path extension', [], [], 'categorical'),
    ('EOD 10 - lymph node', [], [], 'categorical'),
    ('Coding system for EOD', [], [], 'categorical'),
    ('Tumor marker 1', [], [], 'categorical'),
    ('Tumor marker 2', [], [], 'categorical'),
    ('Tumor marker 3', [], [], 'categorical'),
    ('CS Extension', [], [], 'categorical'),
    ('CS Lymph Nodes', [], [], 'categorical'),
    ('CS Mets at DX', [], [], 'categorical'),
    ('CS Site-Specific Factor 1', [], [], 'categorical'),
    ('CS Site-Specific Factor 2', [], [], 'categorical'),
    ('CS Site-Specific Factor 3', [], [], 'categorical'),
    ('CS Site-Specific Factor 4', [], [], 'categorical'),
    ('CS Site-Specific Factor 5', [], [], 'categorical'),
    ('CS Site-Specific Factor 6', [], [], 'categorical'),
    ('CS Site-Specific Factor 25', [], [], 'categorical'),
    ('Derived AJCC T', [], [], 'categorical'),
    ('Derived AJCC N', [], [], 'categorical'),
    ('Derived AJCC M', [], [], 'categorical'),
    ('Derived AJCC Stage Group', [], [], 'categorical'),
    ('Derived SS1977', [], [], 'categorical'),
    ('Derived SS2000', [], [], 'categorical'),
    ('Derived AJCC - flag', [], [], 'categorical'),
    ('CS Version Input Original', [], [], 'categorical'),
    ('CS Version Derived', [], [], 'categorical'),
    ('CS Version Input Current', [], [], 'categorical'),
    ('Site recode ICD-O-3/WHO 2008', [], [], 'categorical'),
    ('Recode ICD-O-2 to 9', [], [], 'categorical'),
    ('Recode ICD-O-2 to 10', [], [], 'categorical'),
    ('ICCC site recode ICD-O-3/WHO 2008', [], [], 'categorical'),
    ('ICCC site rec extended ICD-O-3/ WHO 2008', [], [], 'categorical'),
    ('Behavior recode for analysis', [], [], 'categorical'),
    ('Broad Histology recode', [], [], 'categorical'),
    ('Brain recode', [], [], 'categorical'),
    ('CS Schema v0204', [], [], 'categorical'),
    ('Race recode A', [], [], 'categorical'),
    ('Race recode Y', [], [], 'categorical'),
    ('Origin Recode NHIA', [], [], 'categorical'),
    ('SEER historic stage A', [], [], 'categorical'),
    ('AJCC stage 3rd edition (1988+)', [], [], 'categorical'),
    ('SEER modified AJCC stage 3rd ed (1988+)', [], [], 'categorical'),
    ('SEER Summary Stage 1977 (1995-2000)', [], [], 'categorical'),
    ('SEER Summary Stage 2000 2000 (2001-2003)', [], [], 'categorical'),
    ('First malignant primary indicator', [], [], 'categorical'),
    ('State-county recode', [], [], 'categorical'),
    ('IHS link', [], [], 'categorical'),
    ('Historic SSG 2000 Stage', [], [], 'categorical'),
    ('AYA site recode/WHO 2008', [], [], 'categorical'),
    ('Lymphoma subtype recode/WHO 2008', [], [], 'categorical'),
    ('Primary by International Rules', [], [], 'categorical'),
    ('ER Status Recode Breast Cancer (1990+)', [], [], 'categorical'),
    ('PR Status Recode Breast Cancer (1990+)', [], [], 'categorical'),
    ('CS Schema - AJCC 6th Edition', [], [], 'categorical'),
    ('Cs Site-specific Factor 8', [], [], 'categorical'),
    ('CS Site-Specific Factor 10', [], [], 'categorical'),
    ('CS Site-Specific Factor 11', [], [], 'categorical'),
    ('CS Site-Specific Factor 13', [], [], 'categorical'),
    ('CS Site-Specific Factor 15', [], [], 'categorical'),
    ('CS Site-Specific Factor 16', [], [], 'categorical'),
    ('Lymph-vascular Invasion (2004+)', [], [], 'categorical'),
    ('Insurance Recode (2007+)', [], [], 'categorical'),
    ('Derived AJCC T 7th ed', [], [], 'categorical'),
    ('Derived AJCC N 7th ed', [], [], 'categorical'),
    ('Derived AJCC M 7th ed', [], [], 'categorical'),
    ('Derived AJCC 7 Stage Group', [], [], 'categorical'),
    ('Adjusted AJCC 6th T (1988+)', [], [], 'categorical'),
    ('Adjusted AJCC 6th N (1988+)', [], [], 'categorical'),
    ('Adjusted AJCC 6th M (1988+)', [], [], 'categorical'),
    ('Adjusted AJCC 6th Stage (1988+)', [], [], 'categorical'),
    ('CS Site-Specific Factor 7', [], [], 'categorical'),
    ('CS Site-specific Factor 9', [], [], 'categorical'),
    ('CS Site-Specific Factor 12', [], [], 'categorical'),
    ('Derived HER2 Recode (2010+)', [], [], 'categorical'),
    ('Breast Subtype (2010+)', [], [], 'categorical'),
    ('Lymphoma - Ann Arbor Stage (1983+)', [], [], 'categorical'),
    ('CS mets at DX-bone (2010+)', [], [], 'categorical'),
    ('CS mets at DX-brain (2010+)', [], [], 'categorical'),
    ('CS mets at DX-liver (2010+)', [], [], 'categorical'),
    ('CS mets at DX-lung (2010+)', [], [], 'categorical'),
    ('T value - based on AJCC 3rd (1988-2003)', [], [], 'categorical'),
    ('N value - based on AJCC 3rd (1988-2003)', [], [], 'categorical'),
    ('M value - based on AJCC 3rd (1988-2003)', [], [], 'categorical'),

    # Numerical inputs
    # If 1-n encoding used, encode special codes (unknown, empty) as distinct 1-n vector
    # Encode empty (-1) for all continuous and special codes depending on the variable (see SEER data dictionary)
    ('Age at diagnosis', [(fc.encode_values, [[-1, 999]])], [], 'continuous'),
    ('Year of birth', [(fc.encode_values, [[-1]])], [], 'continuous'),
    ('Year of diagnosis', [(fc.encode_values, [[-1]])], [], 'continuous'),

    ('EOD 10 - positive lymph nodes examined', [(fc.encode_values, [[-1, 90, 95, 97, 98, 99]])], [], 'continuous'),
    ('EOD 10 - number of lymph nodes examined', [(fc.encode_values, [[-1, 90, 95, 96, 97, 98, 99]])], [], 'continuous'),
    ('CS Tumor size', [(fc.encode_values, [[-1, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 888]])], [],
     'continuous'),
    ('Age recode <1 year olds', [(fc.encode_values, [[-1, 99]])], [], 'continuous'),

    # Modified Inputs:
    # Remove info if more than one primary, not applicable for experiments
    ('Sequence number', [(fc.map_values, [{0: 1, 60: 61}])], [], 'categorical'),

    # Target relevant inputs - will be transformed into the target later
    ('SEER cause of death classification', [], [], 'target'),
    ('Survival months', [], [], 'target'),

    # Removed due to irrelevancy:
    ('Patient ID', [], [], 'remove'),
    ('Survival months flag', [], [], 'remove'),
    ('Record number', [], [], 'remove'),
    ('Type of followup expected', [], [], 'remove'),
    # Compound information in one variable
    ('EOD--old 13 digit', [], [], 'remove'),
    ('EOD--old 2 digit', [], [], 'remove'),
    ('EOD--old 4 digit', [], [], 'remove'),

    # Removed due to information added after diagnosis:
    # might contain information about a confirmation after the initial diagnosis.
    ('Diagnostic confirmation', [], [], 'remove'),
    ('Cause of death to SEER site recode', [], [], 'remove'),
    ('SEER other cause of death classification', [], [], 'remove'),
    ('COD to site rec KM', [], [], 'remove'),
    ('Vital status recode (study cutoff used)', [], [], 'remove'),
    ('Total number of in situ/malignant tumors for patient', [], [], 'remove'),
    ('Total number of benign/borderline tumors for patient', [], [], 'remove'),

    # Removed due to treatment information:
    ('RX Summ--surg prim site', [], [], 'remove'),
    ('RX Summ--scope reg LN sur 2003+', [], [], 'remove'),
    ('RX Summ--surg oth reg/dis', [], [], 'remove'),
    ('Number of lymph nodes', [], [], 'remove'),
    ('Reason no cancer-directed surgery', [], [], 'remove'),
    ('Site specific surgery (1983-1997)', [], [], 'remove'),
    ('Scope of lymph node surgery 98-02', [], [], 'remove'),
    ('Surgery to other sites', [], [], 'remove'),
    ('CS EXT/Size Eval', [], [], 'remove'),
    ('CS Nodes Eval', [], [], 'remove'),
    ('CS Mets Eval', [], [], 'remove')
]