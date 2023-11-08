import socket

def get_machine_config():
    if socket.gethostname() == "hostname":
        return {
            'vids_per_gpu': 24,
            'ikea_root': '/path/to/IKEA_ASM_Dataset',
            'epic_root': '/path/to/EPIC-KITCHENS',
            'finegym_root': '/path/to/Fine_GYM',
            'mhad_root': '/path/to/MHAD'
        }
