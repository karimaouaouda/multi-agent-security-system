import csv
import os


class Logger:
    def __init__(self, logfile_path:str, cols:list[str]):
        self.log_path = logfile_path
        self.writer = self.setup_writer()
        self.cols = cols
        self.draw_cols()

    def setup_writer(self):
        if os.path.exists(self.log_path):
            return csv.writer(open(self.log_path, 'w', newline='', encoding='utf-8'))

        with open(self.log_path, 'w', newline='', encoding='utf-8') as csvfile:
            self.writer = csv.writer(csvfile)

    def log(self, values:list):
        self.writer.writerow(values)

    def draw_cols(self):
        #check if size of file is bigger then 0
        if os.path.exists(self.log_path) and os.path.isfile(self.log_path) and os.stat(self.log_path).st_size > 0:
            return True

        self.writer.writerow(self.cols)
