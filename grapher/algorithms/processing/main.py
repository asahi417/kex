""" provide processing instance for `en` or `jp` """


def Processing(language: str = 'en',
               **kwargs):

    if language == 'en':
        from .en import Processing

    elif language == 'jp':
        from .jp import Processing

    else:
        raise ValueError('unknown language: %s not in [`en`, `jp`]' % language)

    return Processing(**kwargs)
