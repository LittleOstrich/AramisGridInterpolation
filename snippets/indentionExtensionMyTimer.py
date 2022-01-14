import inspect
import logging

logger = logging.getLogger(__name__)


def debug(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logger.debug('{i} [{m}]'.format(
        i='.' * indentation_level,
        m=msg
    ))


def foo():
    debug('Hi Mom')
    for i in range(1):
        debug("Now we're cookin")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    foo()