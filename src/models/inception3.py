import warnings
from collections import namedtuple
from typing import Callable
from typing import List
from typing import Optional

import torch
from torchvision.models import Inception3

InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])
InceptionOutputs.__annotations__ = {"logits": torch.Tensor, "aux_logits": Optional[torch.Tensor]}

class ModifiedInception3(Inception3):
    def __init__(
        self,
        num_classes: int = 2,
        aux_logits: bool = True,
        transform_input: bool = False,
        inception_blocks: Optional[List[Callable[..., torch.nn.Module]]] = None,
        init_weights: Optional[bool] = True,
        dropout: float = 0.5,
    ):
        super(ModifiedInception3, self).__init__(
            num_classes=num_classes,
            aux_logits=aux_logits,
            transform_input=transform_input,
            inception_blocks=inception_blocks,
            init_weights=init_weights,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_batches = x.shape[0]
        num_slices = x.shape[1]

        scores = torch.tensor([[0.0, 0.0]] * num_batches)

        x = x.permute(1, 0, 2, 3, 4)
        for cur_x in x:
            cur_x = self._transform_input(cur_x)
            cur_x, aux = self._forward(cur_x)
            aux_defined = self.training and self.aux_logits
            if torch.jit.is_scripting():
                if not aux_defined:
                    warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
                cur_x = InceptionOutputs(cur_x, aux)
            else:
                cur_x = self.eager_outputs(cur_x, aux)
            scores += torch.div(cur_x[0], num_slices)

        # TODO: add sigmoid here and everywhere else
        return scores
