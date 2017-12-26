import sys


def print_utf8(utf8_string):
    sys.stdout.buffer.write(utf8_string.encode('utf-8'))
    sys.stdout.flush()
    print('')

