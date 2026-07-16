class_mapping_full = {
    'hc': 0,
    'CIS': 1,
    'RRMS': 2,
    'SPMS': 3,
    'PPMS': 4,
}

class_mapping_hc_ms = {
    'hc': 0,
    'CIS': 1,
    'RRMS': 1,
    'SPMS': 1,
    'PPMS': 1,
}

class_mapping_hc_cis_ms = {
    'hc': 0,
    'CIS': 1,
    'RRMS': 2,
    'SPMS': 2,
    'PPMS': 2,
}

def get_labels_full(label):
    return label, class_mapping_full

def get_labels_hc_ms(label):
    label_hc_ms = label.copy()
    label_hc_ms[label_hc_ms != 0] = 1  # everything not 0 (HC) becomes 1
    return label_hc_ms, {'hc': 0, 'MS': 1}

def get_labels_hc_cis_ms(label):
    label_hc_cis_ms = label.copy()
    label_hc_cis_ms = label_hc_cis_ms.map(lambda x: x if x in [0, 1] else 2)
    return label_hc_cis_ms, {'hc': 0, 'CIS': 1, 'MS': 2}
