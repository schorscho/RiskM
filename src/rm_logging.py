import logging.config


DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': { 
        'standard': { 
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': { 
        'default': { 
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': { 
        'root': { 
            'handlers': ['default'],
            'level': 'INFO'
        },
        'RiskM': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        },
    } 
}


logging.config.dictConfig(DEFAULT_LOGGING)


logger = logging.getLogger('RiskM')


def time_it(start, end):
    h, r = divmod(end - start, 3600)
    m, s = divmod(r, 60)
    
    return "{:0>2}:{:0>2}:{:06.3f}".format(int(h), int(m), s)


