from time import time

import keras


class TimedStopping(keras.callbacks.Callback):
    """
        Stop training when maximum time has passed.
        Code from:
            https://github.com/keras-team/keras-contrib/issues/87

        Attributes
        ----------
        start_time : float
            time when the training started

        seconds : float
            maximum time before stopping.

        verbose : bool
            verbosity mode.


        Methods
        -------
        on_train_begin(logs)
            method called upon training beginning

        on_epoch_end(epoch, logs={})
            method called after the end of each training epoch
    """

    def __init__(self, seconds=None, verbose=0):
        """
        Parameters
        ----------
        seconds : float
            maximum time before stopping.

        vebose : bool
            verbosity mode
        """

        super(keras.callbacks.Callback, self).__init__()

        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        """
            Method called upon training beginning

            Parameters
            ----------
            logs : dict
                training logs
        """

        self.start_time = time()

    def on_epoch_end(self, epoch, logs={}):
        """
            Method called after the end of each training epoch.
            Checks if the maximum time has passed

            Parameters
            ----------
            epoch : int
                current epoch

            logs : dict
                training logs
        """

        if time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose:
                print('Stopping after %s seconds.' % self.seconds)
