import time


class Progress:

    def __init__(self, logger):
        self.time_start = time.time()
        self.logger = logger

    def start(self):
        self.time_start = time.time()

    def print(self, ex_done, ex_left):
        """
        Prints progress
        :param ex_done:     Number of examples finished since the progress started
        :param ex_left:     Number of examples left
        :return:            None
        """

        # Time calculations.
        time_taken = time.time() - self.time_start
        time_taken_per_ex = time_taken / float(max(1, ex_done))
        time_to_complete = time_taken_per_ex * ex_left

        self.logger.log(f"Processed {ex_done} examples. Total time taken is {beautify(time_taken)}. "
                        f"Avg time per example is {beautify(time_taken_per_ex)}.")
        self.logger.log(f"Remaining {ex_left} examples. Expected Total time taken to complete is "
                        f"{beautify(time_to_complete)}.")


def beautify(time_taken_sec):
    """ Given time taken in seconds it returns a beautified string """

    time_taken_sec = int(time_taken_sec)

    if time_taken_sec < 60:
        return "%r second" % time_taken_sec
    elif 60 <= time_taken_sec < 60 * 60:
        return "%d minutes" % int(time_taken_sec / 60.0)
    elif 60 * 60 <= time_taken_sec < 24 * 60 * 60:
        return "%d hours" % int(time_taken_sec / (60.0 * 60.0))
    elif 24 * 60 * 60 <= time_taken_sec < 30 * 24 * 60 * 60:
        return "%d days" % int(time_taken_sec / (24.0 * 60.0 * 60.0))
    elif 30 * 24 * 60 * 60 <= time_taken_sec < 365 * 24 * 60 * 60:
        return "%d months" % int(time_taken_sec / (30 * 24 * 60 * 60))
    elif 365 * 24 * 60 * 60 <= time_taken_sec:
        months = int((time_taken_sec % (365.0 * 24 * 60.0 * 60.0)) / (30.0 * 24.0 * 60.0 * 60.0))
        return "%d years %d months" % (int(time_taken_sec / (365.0 * 24.0 * 60.0 * 60.0)), months)


def elapsed_from_str(time_from):
    """ Given a time from, create timestep using the current time step"""

    return beautify(time.time() - time_from)
