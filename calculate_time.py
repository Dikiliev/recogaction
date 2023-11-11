import datetime


class TimeCalculator:

    def start(self):
        self.start_time = datetime.datetime.now()

    def end(self):
        self.end_time = datetime.datetime.now()


    def get_passed_time(self):
        return str((self.end_time - self.start_time).seconds) + ' seconds'