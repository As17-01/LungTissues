from typing import Callable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import torch
import torch.nn.functional as f
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck


class ModifiedResNet18(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
        layers: List[int] = [2, 2, 2, 2],
        num_classes: int = 2,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
    ):
        super(ModifiedResNet18, self).__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        slide_size = x.shape[1]

        scores = torch.tensor([[0.0, 0.0]] * batch_size)

        x = x.permute(1, 0, 2, 3, 4)
        for cur_x in x:
            cur_x = self._forward_impl(cur_x)
            cur_x = f.sigmoid(cur_x)
            scores += torch.div(cur_x, slide_size)
        return scores
