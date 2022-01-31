import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from utils.opencv_plotting import BurstSRVis
import torch
import cv2
import numpy as np
from utils.postprocessing_functions import SimplePostProcess
from datasets.synthetic_burst_val_set import SyntheticBurstVal


def visualize_results():
    """ Visualize the results on the SyntheticBurst validation set.
    """

    vis = BurstSRVis(boundary_ignore=40)
    process_fn = SimplePostProcess(return_np=True)

    dataset = SyntheticBurstVal('PATH_TO_SyntheticBurstVal')

    for idx in range(len(dataset)):
        burst, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']

        pred_path = '{}/{}.png'.format('PATH_TO_RESULTS', burst_name)
        pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        pred = torch.from_numpy(pred.astype(np.float32) / 2 ** 14).permute(2, 0, 1)

        pred = process_fn.process(pred, meta_info)
        data = [{'images': [pred, ],
                 'titles': ['Pred', ]}]
        cmd = vis.plot(data)

        if cmd == 'stop':
            return


if __name__ == '__main__':
    visualize_results()
