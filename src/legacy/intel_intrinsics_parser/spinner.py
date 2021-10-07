# Copyright 2021 Marcos Horro
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import threading
import sys


class Spinner:
    busy = False
    delay = 0.075
    init_msg = ""
    end_msg = ""

    @staticmethod
    def spinning_cursor():
        while True:
            for cursor in "|/-\\":
                yield cursor

    def __init__(self, init_msg="Loading task...", end_msg="OK", delay=None):
        self.spinner_generator = self.spinning_cursor()
        self.init_msg = init_msg
        self.end_msg = end_msg
        if delay and float(delay):
            self.delay = delay

    def spinner_task(self):
        sys.stdout.write(self.init_msg)
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write("\b")
            sys.stdout.flush()
        sys.stdout.write(" \b")
        sys.stdout.write("\r" + " " * len(self.init_msg))
        sys.stdout.write("\r%s\n" % self.end_msg)

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False
