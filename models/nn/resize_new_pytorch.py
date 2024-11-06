from typing import Optional
from typing import Any, List

import numpy as np
import torch as torch
import torch.nn.functional as F


def round_n_bits(x, decimal_bits):
    scaled_value = x * (2 ** decimal_bits)
    rounded_value = np.floor(scaled_value)  # very important
    result = rounded_value / (2 ** decimal_bits)
    return result


def result_four_point(p, w1, w2):
    w_row_left = w1
    w_row_right = w2
    w_column_up = w_row_left
    w_column_down = w_row_right
    # print(sum(w_column_up))
    # print(sum(w_column_down))

    x_column_left = torch.sum(w_row_left * p[:, :, 0:4], dim=2)
    x_column_right = torch.sum(w_row_right * p[:, :, 1:5], dim=2)
    # print(x_column_left)
    # print(x_column_right)

    r1 = torch.sum(x_column_left[:, 0:4] * w_column_up, dim=1)
    r2 = torch.sum(x_column_right[:, 0:4] * w_column_up, dim=1)
    r3 = torch.sum(x_column_left[:, 1:5] * w_column_down, dim=1)
    r4 = torch.sum(x_column_right[:, 1:5] * w_column_down, dim=1)
    # print(r1)
    return r1, r2, r3, r4


def resize_new(
        x: torch.Tensor,
        size: Optional[Any] = None,
        scale_factor: Optional[List[float]] = None,
        mode: str = "bicubic",
        align_corners: Optional[bool] = False,
) -> torch.Tensor:
    device = x.device
    s = x.shape
    # print(s)

    # output image
    pout = torch.zeros([s[0] * scale_factor[0], s[1] * scale_factor[1], ((s[2]) * scale_factor[2]),
                        ((s[3]) * scale_factor[3])])
    # print(pout.shape)

    # padding, replicate
    pin_append = F.pad(x, (2, 2, 2, 2), 'replicate')
    # print(pin_append)

    # 4 weights
    w_row_left = torch.tensor([-0.0234375, 0.2265625, 0.8671875, -0.0703125], device=device)
    w_row_right = torch.tensor([-0.0703125, 0.8671875, 0.2265625, -0.0234375], device=device)
    # print(w_row_left)

    # unfold, 5x5 window
    patches = pin_append.unfold(2, 5, 1).unfold(3, 5, 1)
    p = patches.reshape(-1, 5, 5)

    r1, r2, r3, r4 = result_four_point(p, w_row_left, w_row_right)
    r1 = r1.view(s[0], s[1], s[2], s[3])
    r2 = r2.view(s[0], s[1], s[2], s[3])
    r3 = r3.view(s[0], s[1], s[2], s[3])
    r4 = r4.view(s[0], s[1], s[2], s[3])

    pout[:, :, ::2, ::2] = r1
    pout[:, :, ::2, 1::2] = r2
    pout[:, :, 1::2, ::2] = r3
    pout[:, :, 1::2, 1::2] = r4
    return pout


if __name__ == '__main__':
    pixel_in1 = torch.tensor([
        [[[1, 1, 1, 1, 1],
          [1, 2, 1, 1, 1],
          [1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1]]]
    ], dtype=torch.float32)
    pixel_in3 = torch.tensor([
        [[[1, 1, 3, 1],
          [1, 2, 1, 1],
          [1, 9, 1, 2],
          [1, 1, 1, 1]],
         [[1, 1, 5, 1],
          [1, 2, 1, 1],
          [1, 1, 8, 1],
          [1, 4, 1, 1]]]
    ], dtype=torch.float64)
    d = resize_new(pixel_in1, scale_factor=[1, 1, 2, 2], mode='bicubic', align_corners=False)
    # print(d)
    # p = pixel_in1.unfold(2, 2, 1).unfold(3, 2, 1)
    # print(p.shape)
    # p1 = p.permute(0, 1, 2, 4, 3, 5).reshape(-1, 2, 2)
    # p1 = p.reshape(-1, 2, 2)
    # print(p1.shape)
    # print(p)
    # print(p1)
    # print(pixel_in1[:,:,::2,::2])
