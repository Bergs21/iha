import torch

if torch.cuda.is_available():
    print('GPU kullanılıyor')
else:
    print('GPU kullanılabilir değil')
