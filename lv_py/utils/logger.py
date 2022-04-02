import logging
import os
import datetime

class Logger():
    def __init__(self, path, use_stdout=True):

        if not os.path.exists(path):
            os.makedirs(path)


        now = datetime.datetime.now()
        time = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
        path = f"{path}/{time}.log"

        open(path, "w").close()

        self.use_stdout = use_stdout

        self.logger = logging.getLogger(__name__)
        self.formatter = logging.Formatter(
            fmt='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')

        # File logger
        self.file_handler = logging.FileHandler(path, "a+")
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

        # Stdout logger
        if self.use_stdout:
            self.stdout_handler = logging.StreamHandler()
            self.stdout_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.stdout_handler)

        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def info(self, txt):
        self.logger.info(txt)

    def dispose(self):
        self.file_handler.close()
        if self.use_stdout:
            self.stdout_handler.close()

    def __del__(self):
        self.dispose()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name = "Unknown"):
        self.reset()
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count

if __name__ == "__main__":
    logger = Logger(use_stdout=True)
    logger.info("Hello World")
    logger.info("Hello World")
    logger.info("Hello World")
    logger.info("Hello World")
