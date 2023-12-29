import logging


class Logger:

    def __init__(self, save_dir, fname):

        logging.basicConfig(filename=f"{save_dir}/{fname}",
                            filemode="a",
                            format='%(name)s, %(levelname)s: %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        self.logger = logging.getLogger("Main")

    def log(self, msg, also_stdout=True):
        if also_stdout:
            print(f"Main: Msg: {msg}")
        self.logger.info(msg)
