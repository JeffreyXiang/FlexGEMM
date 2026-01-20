from ...utils import get_autotune_config


USE_ON_THE_FLY_WEIGHT_TRANSPOSE = True


autotune_config = get_autotune_config(
    platform={
        'cuda': [
            {'B1': 128, 'B2': 256, 'BK': 64, 'num_stages': 3, 'num_warps': 8, 'smem_neighbor': True},
            {'B1': 64,  'B2': 256, 'BK': 32, 'num_stages': 4, 'num_warps': 4, 'smem_neighbor': True},
            {'B1': 128, 'B2': 128, 'BK': 32, 'num_stages': 4, 'num_warps': 4, 'smem_neighbor': True},
            {'B1': 128, 'B2': 64,  'BK': 32, 'num_stages': 4, 'num_warps': 4, 'smem_neighbor': True},
            {'B1': 64,  'B2': 128, 'BK': 32, 'num_stages': 4, 'num_warps': 4, 'smem_neighbor': True},
            {'B1': 128, 'B2': 32,  'BK': 32, 'num_stages': 4, 'num_warps': 4, 'smem_neighbor': True},
            {'B1': 64,  'B2': 32,  'BK': 32, 'num_stages': 5, 'num_warps': 2, 'smem_neighbor': True},
            {'B1': 32,  'B2': 64,  'BK': 32, 'num_stages': 5, 'num_warps': 2, 'smem_neighbor': True},
        ],
    },
    device={
        'A100': [
            {'B1': 256, 'B2': 128, 'BK': 64, 'num_stages': 3, 'num_warps': 8, 'smem_neighbor': True},
            {'B1': 256, 'B2': 128, 'BK': 64, 'num_stages': 3, 'num_warps': 8, 'smem_neighbor': False},
            {'B1': 256, 'B2': 128, 'BK': 32, 'num_stages': 4, 'num_warps': 8, 'smem_neighbor': True},
            {'B1': 256, 'B2': 128, 'BK': 32, 'num_stages': 4, 'num_warps': 8, 'smem_neighbor': False},
            {'B1': 256, 'B2': 64 , 'BK': 32, 'num_stages': 4, 'num_warps': 4, 'smem_neighbor': True},
            {'B1': 256, 'B2': 64 , 'BK': 32, 'num_stages': 4, 'num_warps': 4, 'smem_neighbor': False},
            {'B1': 128, 'B2': 256, 'BK': 64, 'num_stages': 3, 'num_warps': 8, 'smem_neighbor': True},
            {'B1': 128, 'B2': 256, 'BK': 64, 'num_stages': 3, 'num_warps': 8, 'smem_neighbor': False},
            {'B1': 128, 'B2': 256, 'BK': 32, 'num_stages': 4, 'num_warps': 8, 'smem_neighbor': True},
            {'B1': 128, 'B2': 256, 'BK': 32, 'num_stages': 4, 'num_warps': 8, 'smem_neighbor': False},
            {'B1': 128, 'B2': 128, 'BK': 64, 'num_stages': 4, 'num_warps': 8, 'smem_neighbor': True},
            {'B1': 128, 'B2': 128, 'BK': 64, 'num_stages': 4, 'num_warps': 8, 'smem_neighbor': False},
            {'B1': 128, 'B2': 128, 'BK': 32, 'num_stages': 4, 'num_warps': 4, 'smem_neighbor': True},
            {'B1': 128, 'B2': 128, 'BK': 32, 'num_stages': 4, 'num_warps': 4, 'smem_neighbor': False},
        ],
        'H100': [
            {'B1': 256, 'B2': 128, 'BK': 64, 'num_stages': 4, 'num_warps': 8},
            {'B1': 256, 'B2': 128, 'BK': 64, 'num_stages': 4, 'num_warps': 8},
            # {'B1': 256, 'B2': 128, 'BK': 32, 'num_stages': 4, 'num_warps': 8},
            # {'B1': 128, 'B2': 256, 'BK': 64, 'num_stages': 4, 'num_warps': 8},
            # {'B1': 128, 'B2': 256, 'BK': 32, 'num_stages': 4, 'num_warps': 8},
            # {'B1': 256, 'B2': 64,  'BK': 64, 'num_stages': 4, 'num_warps': 4},
            # {'B1': 256, 'B2': 64,  'BK': 32, 'num_stages': 4, 'num_warps': 4},
            # {'B1': 64,  'B2': 256, 'BK': 64, 'num_stages': 4, 'num_warps': 4},
            # {'B1': 64,  'B2': 256, 'BK': 32, 'num_stages': 4, 'num_warps': 4},
            # {'B1': 128, 'B2': 128, 'BK': 64, 'num_stages': 4, 'num_warps': 4},
            # {'B1': 128, 'B2': 128, 'BK': 32, 'num_stages': 4, 'num_warps': 4},
            # {'B1': 128, 'B2': 64,  'BK': 32, 'num_stages': 4, 'num_warps': 4},
            # {'B1': 128, 'B2': 64,  'BK': 32, 'num_stages': 4, 'num_warps': 2},
            # {'B1': 64,  'B2': 128, 'BK': 32, 'num_stages': 4, 'num_warps': 4},
            # {'B1': 64,  'B2': 128, 'BK': 32, 'num_stages': 4, 'num_warps': 2},
            # {'B1': 64,  'B2': 64,  'BK': 64, 'num_stages': 4, 'num_warps': 2},
            # {'B1': 64,  'B2': 64,  'BK': 32, 'num_stages': 4, 'num_warps': 2},
        ],
    }
)
