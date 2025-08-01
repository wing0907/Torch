import torch

# Pytorch 
print('Pytorch 버전 :', torch.__version__)

# CUDA 사용 여부 확인
cuda_available = torch.cuda.is_available()
print('CUDA 사용 가능 여부 :', cuda_available)

# 사용 가능 GPU 개수 확인
gpu_count = torch.cuda.device_count()
print('사용 가능 GPU 개수 :', gpu_count)

if cuda_available:
    # 현재 사용중인 GPU 장치 확인
    current_device = torch.cuda.current_device()
    print('현재 사용중인 GPU 장치 ID l', current_device)
    print('현재 GPU 이름 :', torch.cuda.get_device_name(current_device))
else:
    print("GPU 없음")

# CUDA 버전 확인
print('CUDA 버전 :', torch.version.cuda)

# CUDNN 버전 확인
cudnn_version = torch.backends.cudnn.version()
if cudnn_version is not None:
    print("cuDNN 버전 :", cudnn_version)
else:
    print('cuDNN 없음')

# Pytorch 버전 : 2.7.1+cu118
# CUDA 사용 가능 여부 : True
# 사용 가능 GPU 개수 : 1
# 현재 사용중인 GPU 장치 ID l 0
# 현재 GPU 이름 : NVIDIA GeForce RTX 3050
# CUDA 버전 : 11.8
# cuDNN 버전 : 90100