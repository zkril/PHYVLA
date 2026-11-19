import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download

snapshot_download(
      repo_id='yifengzhu-hf/LIBERO-datasets',
      repo_type='dataset',
      local_dir='./data',
      allow_patterns='libero_object/*',
      local_dir_use_symlinks=False
)
print('libero_object downloaded!')