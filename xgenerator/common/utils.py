import logging
import sys

import torch


def try_cuda(pytorch_obj):
    import torch.cuda
    try:
        disabled = torch.cuda.disabled
    except:
        disabled = False
    if torch.cuda.is_available() and not disabled:
        return pytorch_obj.cuda()
    else:
        return pytorch_obj


class MyLogger(logging.Logger):
    def __init__(self, name: str, level: str, filename: str):
        """A logger for logging experiments.
        
        Args:
            name (str): A name of this logger.
            level (str): A level to log.
            filename (str): A file for logging experiments.
        """
        super().__init__(name, level)
        self._formatter = logging.Formatter("%(asctime)-15s %(message)s", None, "%")

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self._formatter)
        stream_handler.setLevel(level)
        self.addHandler(stream_handler)

        filehandler = logging.FileHandler(filename)
        filehandler.setFormatter(self._formatter)
        filehandler.setLevel(level)
        self.addHandler(filehandler)
