'''Format time to string'''

from datetime import timedelta


def time2str(t):
    '''
    :param t: time in seconds
    :return: string of the format h:mm:ss
    '''

    hours = str(timedelta(seconds=t)).split(':')[0]
    minutes = str(timedelta(seconds=t)).split(':')[1]
    seconds = str(timedelta(seconds=t)).split(':')[2].split('.')[0]

    return '{}:{}:{}'.format(hours, minutes, seconds)
